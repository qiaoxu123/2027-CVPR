"""Static-only CityNav goal localization: Prior -> Rules -> Typed GAT.

No RGB, UAV pose, explored area, teacher trajectory, target image, or future
observation is included in forward(). Human goal is a TRAINING LABEL only.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from .geometry import grid_centers, normalize_xy
from .model import DenseGoalField
from .semantic_geometry import canonical, prior_and_rule_fields
from .semantic_graph import TypedGeometryGAT, prepare_semantic_graph


class StaticLocalizationDataset(Dataset):
    """Reads existing independent CityNav JSONL, optional named road polylines.

    road_json, when given, maps map names to items:
    { "cambridge_block_3": [
      {"id":"osm-way-9","name":"Main Road",
       "polyline":[[x,y],...], "coordinate_system":"world"} ]}
    Other accepted coordinate_system is 'normalized'. There is no inference
    from a candidate GT point or hidden full-city image.
    """

    def __init__(
        self, jsonl_path, *, grid_size=28, max_landmarks=16, max_roads=12,
        max_relations=8, roads_json=None, prior_sigma_m=25.,
        rule_sigma_m=25.,
        deduplicate_queries=True,
    ):
        raw_records = [
            json.loads(line)
            for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        # Static maps/instructions do not change across consecutive human
        # trajectory poses. Keep ONE record per language query, otherwise
        # longer CityFlight paths get artificially higher evaluation weight.
        self.records = []
        seen_queries = set()
        for record in raw_records:
            # The observable static query is map + instruction. Neither GT
            # coordinates nor annotation IDs may give repeated text extra weight.
            # Preserve the first source record deterministically.
            query_id = (record.get("map_name", ""), record["instruction"])
            if deduplicate_queries and query_id in seen_queries:
                continue
            seen_queries.add(query_id)
            self.records.append(record)
        if not self.records:
            raise ValueError("empty CityNav JSONL")
        self.roads_by_map = (
            json.loads(Path(roads_json).read_text(encoding="utf-8")) if roads_json else {}
        )
        self.grid_size = grid_size
        self.max_landmarks = max_landmarks
        self.max_roads = max_roads
        self.max_relations = max_relations
        self.prior_sigma_m = prior_sigma_m
        self.rule_sigma_m = rule_sigma_m
        self.grid_xy = grid_centers(grid_size).numpy()
        self.capacity = max_landmarks + max_roads + max_relations
        # Cache belongs to this dataset, not a global decorator retaining every
        # dataset instance. Graphs are invariant over epochs and target labels.
        self._prepared = lru_cache(maxsize=None)(self._prepare)

    def __len__(self):
        return len(self.records)

    @staticmethod
    def normalized_roads(roads, bounds):
        result = []
        for road in roads:
            system = road.get("coordinate_system", None)
            if system not in ("normalized", "world"):
                raise ValueError("road coordinate_system must be explicit")
            polyline = np.asarray(road["polyline"], dtype=np.float32)
            if system == "world":
                polyline = normalize_xy(polyline, bounds)
            if np.any(polyline < -1e-4) or np.any(polyline > 1+1e-4):
                raise ValueError("road coordinates out of declared map bounds")
            result.append({
                "id": road.get("id", ""), "name": road.get("name", ""),
                "polyline": polyline.tolist(),
                "osm_node_ids": road.get("osm_node_ids", []),
            })
        return result

    def _prepare(self, idx):
        record = self.records[idx]
        bounds = record.get("world_bounds")
        if bounds is None:
            raise ValueError("CityNav record missing public world_bounds; re-run converter")
        refs = record.get("landmarks", [])[:self.max_landmarks]
        # Select instruction-mentioned named roads FIRST. Unmentioned roads are
        # sorted by proximity to referenced landmark centers, not target XY.
        all_roads = self.roads_by_map.get(record.get("map_name", ""), [])
        sentence = " " + canonical(record["instruction"]) + " "
        ref_centers = [
            np.asarray(lm["contour"], np.float32).mean(0)
            for lm in refs if len(lm.get("contour", [])) >= 3
        ]
        def road_priority(road):
            named = canonical(road.get("name", ""))
            mentioned = bool(named and (" " + named + " ") in sentence)
            poly = np.asarray(road.get("polyline", []), np.float32)
            if poly.ndim != 2 or len(poly) < 2:
                near = float("inf")
            else:
                bounds_lo = np.asarray(bounds[:2], np.float32)
                bounds_extent = np.asarray(bounds[2:], np.float32) - bounds_lo
                if road.get("coordinate_system") == "world":
                    xy = (poly - bounds_lo) / bounds_extent
                else:
                    xy = poly
                center = xy.mean(0)
                near = min(
                    (float(np.linalg.norm(center - c)) for c in ref_centers),
                    default=0.,
                )
            return (0 if mentioned else 1, near, str(road.get("id", "")))
        chosen_roads = sorted(all_roads, key=road_priority)[:self.max_roads]
        roads = self.normalized_roads(chosen_roads, bounds)
        graph = prepare_semantic_graph(
            refs, roads, record["instruction"], bounds,
            max_landmarks=self.max_landmarks, max_roads=self.max_roads,
            max_relations=self.max_relations,
        )
        xymin = np.asarray(bounds[:2], dtype=np.float32)
        spans = np.asarray(bounds[2:], dtype=np.float32) - xymin
        world_grid = xymin[None] + self.grid_xy * spans[None]
        # ALL named CityRefer referenced *surfaces* contribute to the prior:
        # building, road-surface, footpath, parking, bridge, ground, etc.
        # Only unreferenced verified centerlines are excluded.
        referenced_entities = [e for e in graph["entities"] if e["kind"] != "road"]
        prior, _, _ = prior_and_rule_fields(
            world_grid, referenced_entities, [],
            sigma_m=self.prior_sigma_m,
        )
        # Preserve an explicit buildings-only control on the SAME queries,
        # grid and ground truth, to quantify gains from newly restored types.
        building_entities = [
            e for e in referenced_entities if e["kind"] == "building"
        ]
        prior_buildings, _, _ = prior_and_rule_fields(
            world_grid, building_entities, [], sigma_m=self.prior_sigma_m
        )
        _, rules, relation_fields = prior_and_rule_fields(
            world_grid, graph["entities"], graph["relations"],
            sigma_m=self.rule_sigma_m,
        )
        if len(graph["relations"]) > self.max_relations:
            raise ValueError("too many relations")
        rel_padded = np.zeros((len(world_grid), self.max_relations), np.float32)
        if relation_fields.shape[1]:
            rel_padded[:, :relation_fields.shape[1]] = relation_fields
        return graph, prior, prior_buildings, rules, rel_padded

    def __getitem__(self, idx):
        rec = self.records[idx]
        graph, prior, prior_buildings, rules, rel_fields = self._prepared(idx)
        return {
            "text": rec["instruction"],
            "node_names": graph["node_names"],
            "nodes": torch.from_numpy(graph["nodes"]),
            "node_types": torch.from_numpy(graph["node_types"]),
            "node_valid": torch.from_numpy(graph["node_mask"]),
            "edges": torch.from_numpy(graph["edges"]),
            "adjacency": torch.from_numpy(graph["adjacency"]),
            "prior": torch.from_numpy(prior),
            "prior_buildings": torch.from_numpy(prior_buildings),
            "rules": torch.from_numpy(rules),
            "relation_fields": torch.from_numpy(rel_fields),
            "grid_xy": torch.from_numpy(self.grid_xy),
            "target_xy": torch.tensor(rec["target_xy"], dtype=torch.float32),
            "map_sizes": torch.tensor(rec["physical_map_sizes"], dtype=torch.float32),
            "relation_count": len(graph["relations"]),
            "road_relation_count": sum(int(any(graph["entities"][idx]["kind"] == "road" for group in r.anchors for idx in group)) for r in graph["relations"]),
            "road_count": sum(int(e["kind"] == "road") for e in graph["entities"]),
            "surface_count": sum(
                int(e["kind"] != "road") for e in graph["entities"]
            ),
            "nonbuilding_surface_count": sum(
                int(e["kind"] not in ("road", "building"))
                for e in graph["entities"]
            ),
            "edge_count": int(graph["adjacency"].sum()),
            "episode_id": rec.get("episode_id", ""),
        }


def collate_static(items):
    batch = {}
    for name in items[0]:
        batch[name] = (
            [s[name] for s in items] if name in ("text", "node_names", "episode_id")
            else torch.stack([s[name] for s in items])
            if isinstance(items[0][name], torch.Tensor)
            else torch.tensor([s[name] for s in items])
        )
    return batch


class StaticGoalLocalizer(nn.Module):
    """Geometry/rules fixed ablations + trainable typed GAT residual.

    The output is a global [B,G^2] distribution for ALL candidates, not a
    local neighborhood or an oracle-proposed set of positive candidates.
    """

    def __init__(self, hidden=128, text_dim=768, heads=4, relations=8, dropout=.1,
                 rule_weight=1.0):
        super().__init__()
        self.gnn = TypedGeometryGAT(
            hidden=hidden, text_dim=text_dim, heads=heads, dropout=dropout
        )
        self.query = nn.Sequential(nn.Linear(10, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.instruction = nn.Sequential(nn.Linear(text_dim, hidden), nn.LayerNorm(hidden))
        self.relation_proj = nn.Sequential(
            nn.Linear(relations, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.graph_head = nn.Sequential(
            nn.LayerNorm(hidden*3 + 2), nn.Linear(hidden*3 + 2, hidden),
            nn.GELU(), nn.Linear(hidden, 1),
        )
        self.graph_scale = nn.Parameter(torch.tensor(0.10))
        self.rule_weight = float(rule_weight)

    def _baseline(self, batch, variant):
        prior = batch["prior"]
        if variant == "prior":
            return prior
        if variant == "building_prior":
            return batch.get("prior_buildings", prior)
        return prior + self.rule_weight * batch["rules"]

    def forward(self, batch, *, variant="gat", instruction_emb=None, node_emb=None):
        if variant in ("building_prior", "prior", "rules"):
            return self._baseline(batch, variant)
        if variant != "gat":
            raise ValueError(f"unsupported static variant: {variant}")
        if instruction_emb is None or node_emb is None:
            raise ValueError("trained graph variant requires frozen text embeddings")
        _b, q, _ = batch["grid_xy"].shape
        h = self.gnn(
            batch["nodes"], node_emb, batch["node_types"],
            batch["adjacency"], batch["edges"], batch["node_valid"],
        )
        valid = batch["node_valid"]
        query = self.query(DenseGoalField.positional_features(batch["grid_xy"]))
        query = query + self.instruction(instruction_emb)[:, None]
        masked_logits = torch.matmul(query, h.transpose(1, 2)) / query.shape[-1]**.5
        masked_logits = masked_logits.masked_fill(
            ~valid[:, None], torch.finfo(masked_logits.dtype).min
        )
        attention = torch.softmax(masked_logits.float(), dim=-1).to(h.dtype)
        attended = torch.matmul(attention, h)
        relation_type_mask = valid & (batch["node_types"] == 2)
        denom = relation_type_mask.sum(dim=-1, keepdim=True).clamp_min(1)
        relation_context = (h * relation_type_mask.unsqueeze(-1)).sum(1) / denom
        rel_map = self.relation_proj(batch["relation_fields"])
        fused = torch.cat((
            query, attended, relation_context[:, None].expand(-1, q, -1) + rel_map,
            batch["prior"].unsqueeze(-1), batch["rules"].unsqueeze(-1)
        ), dim=-1)
        residual = self.graph_head(fused).squeeze(-1)
        return self._baseline(batch, "rules") + torch.tanh(self.graph_scale) * residual


def cross_entropy_goal(logits, target_xy, grid_size=28, sigma_cells=.9):
    from .losses import goal_distribution

    probs = goal_distribution(target_xy, grid_size, sigma_cells=sigma_cells)
    return -(probs * F.log_softmax(logits.float(), dim=-1)).sum(-1).mean()


@torch.no_grad()
def static_recall(logits, target_xy, map_sizes, grid_xy, radius_m=20., ranks=(1, 4, 8)):
    top = logits.topk(min(max(ranks), logits.shape[-1]), dim=-1).indices
    xy = grid_xy.gather(1, top.unsqueeze(-1).expand(-1, -1, 2))
    distance_m = ((xy - target_xy[:, None]) * map_sizes[:, None]).norm(dim=-1)
    result = {
        "count": len(logits),
        "NE_sum_m": distance_m[:, 0].sum().item(),
    }
    for k in ranks:
        result[f"R@{k}/20m_count"] = int((distance_m[:, :k].min(dim=-1).values <= radius_m).sum().item())
    return result
