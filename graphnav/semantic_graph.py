"""Typed relation graph: one building polygon = one node; one road polyline = one node.

Road-road edges represent actual connected/intersecting polylines (or explicit
OSM topology in a future adapter), NOT geographical k-NN. Relation phrases
are first-class nodes. Features refer to a public map only.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .graph import EdgeGATv2
from .semantic_geometry import (
    EDGE_TYPES, NODE_TYPES, RELATIONS, build_entities,
    geometry_pair_distance, parse_spatial_relations, polygon_center,
)


def _physical_node_geometry(entity, bounds):
    xy = np.asarray(entity["geometry_m"], np.float32)
    low, span = np.asarray(bounds[:2], np.float32), np.asarray(bounds[2:], np.float32) - np.asarray(bounds[:2], np.float32)
    center = xy.mean(axis=0)
    dif = xy.max(axis=0) - xy.min(axis=0)
    diffs = np.diff(xy, axis=0)
    length = float(np.linalg.norm(diffs, axis=-1).sum())
    if entity["kind"] == "building":
        length += float(np.linalg.norm(xy[-1] - xy[0]))
        area = 0.5 * abs(float(
            (xy[:, 0] * np.roll(xy[:, 1], -1) - xy[:, 1] * np.roll(xy[:, 0], -1)).sum()
        ))
    else:
        area = 0.
    axis = (xy[-1] - xy[0]) if entity["kind"] == "road" else dif.copy()
    axis /= max(float(np.linalg.norm(axis)), 1e-6)
    return np.array([
        *((center-low)/span), *(dif/span), length/400.,
        area/(400.**2), *axis, float(entity["kind"] == "building"),
        min(len(xy), 100) / 100.,
    ], np.float32)


def prepare_semantic_graph(
    reference_landmarks, roads, instruction, world_bounds,
    *,
    max_landmarks=16, max_roads=12, max_relations=8,
    near_m=45., road_connect_tolerance_m=1.,
):
    """Pack typed nodes and geometry edges, preserving name/contour identity.

    Roads are optional; 'along Road A' is only grounded if Road A's polyline
    exists. A missing road CANNOT be replaced with an invented straight line.
    """
    references = reference_landmarks[:max_landmarks]
    roads = roads[:max_roads]
    entities = build_entities(references, roads, world_bounds)
    relations = parse_spatial_relations(instruction, entities)[:max_relations]
    n_entity = len(entities)
    capacity = max_landmarks + max_roads + max_relations
    n_nodes = n_entity + len(relations)
    # Use a valid null node if the instruction mentions no mapped entities.
    size = max(capacity, 1)
    feature = np.zeros((size, 10), dtype=np.float32)
    types = np.zeros(size, dtype=np.int64)
    mask = np.zeros(size, dtype=bool)
    center_m = np.zeros((size, 2), dtype=np.float32)
    names = [""] * size
    adj = np.zeros((size, size), dtype=bool)
    e_feat = np.zeros((size, size, 8 + len(EDGE_TYPES)), dtype=np.float32)
    anchor_pairs = np.zeros((size, size), dtype=bool)
    valid_node_count = max(n_nodes, 1)
    if not n_nodes:
        mask[0] = True
        adj[0, 0] = True
        e_feat[0, 0, 8 + EDGE_TYPES["self"]] = 1.

    for i, ent in enumerate(entities):
        feature[i] = _physical_node_geometry(ent, world_bounds)
        types[i] = NODE_TYPES[ent["kind"]]
        mask[i] = True
        center_m[i] = polygon_center(ent["geometry_m"])
        names[i] = ent["name"]
    for r_idx, rel in enumerate(relations):
        idx = n_entity + r_idx
        types[idx] = NODE_TYPES["relation"]
        mask[idx] = True
        names[idx] = rel.kind
        anchors = [a for group in rel.anchors for a in group]
        if anchors:
            center_m[idx] = np.mean([center_m[a] for a in anchors], axis=0)
            feature[idx, :2] = (
                center_m[idx] - np.asarray(world_bounds[:2], np.float32)
            ) / (np.asarray(world_bounds[2:], np.float32) - np.asarray(world_bounds[:2], np.float32))

    def connect(i, j, edge_kind, boundary_distance=0.):
        delta = center_m[j] - center_m[i]
        dist = float(np.linalg.norm(delta))
        normalized = delta / max(dist, 1e-6)
        e_feat[i, j, :8] = [
            delta[0]/400., delta[1]/400., dist/400.,
            boundary_distance/400., normalized[0], normalized[1],
            float(edge_kind == "roads_connected"), 1.,
        ]
        e_feat[i, j, 8+EDGE_TYPES[edge_kind]] = 1.
        adj[i, j] = True

    for i in range(n_nodes):
        connect(i, i, "self")
    for i in range(n_entity):
        for j in range(i + 1, n_entity):
            x, y = entities[i], entities[j]
            is_road_x, is_road_y = x["kind"] == "road", y["kind"] == "road"
            d = geometry_pair_distance(
                x["geometry_m"], y["geometry_m"], is_road_x, is_road_y
            )
            if is_road_x and is_road_y:
                # Reliable OSM topology is shared node identity. Without
                # node IDs, only actual shared endpoints may infer connection.
                ids_x, ids_y = set(x["osm_node_ids"]), set(y["osm_node_ids"])
                if ids_x and ids_y:
                    connected = bool(ids_x & ids_y)
                else:
                    endpoints_x = [x["geometry_m"][0], x["geometry_m"][-1]]
                    endpoints_y = [y["geometry_m"][0], y["geometry_m"][-1]]
                    connected = min(
                        float(np.linalg.norm(a - b))
                        for a in endpoints_x for b in endpoints_y
                    ) <= road_connect_tolerance_m
                kind = ("roads_connected" if connected else
                        ("roads_crossing" if d <= road_connect_tolerance_m else None))
            elif is_road_x != is_road_y:
                kind = "building_road" if d <= near_m else None
            else:
                kind = ("buildings_overlap" if d <= 1e-5 else
                        ("buildings_near" if d <= near_m else None))
            if kind:
                connect(i, j, kind, d)
                connect(j, i, kind, d)
    for r_idx, rel in enumerate(relations):
        rid = n_entity + r_idx
        for group in rel.anchors:
            for anchor in group:
                connect(rid, anchor, "relation_anchor")
                connect(anchor, rid, "anchor_relation")
                anchor_pairs[rid, anchor] = True
    return {
        "nodes": feature, "node_types": types, "node_mask": mask,
        "edges": e_feat, "adjacency": adj, "relation_links": anchor_pairs,
        "node_names": names,
        "entities": entities, "relations": relations,
        "valid_node_count": valid_node_count,
        "edge_type_count": {kind: int(adj.sum()) for kind in ()},
    }


class TypedGeometryGAT(nn.Module):
    """Edge-condition GATv2 on typed relations, with frozen BERT name inputs."""

    def __init__(self, hidden=128, text_dim=768, heads=4, edge_dim=None, dropout=.1):
        super().__init__()
        edge_dim = edge_dim or 8 + len(EDGE_TYPES)
        self.node_geometry = nn.Sequential(nn.Linear(10, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.name_projection = nn.Sequential(nn.Linear(text_dim, hidden), nn.LayerNorm(hidden))
        self.type_embedding = nn.Embedding(len(NODE_TYPES), hidden)
        self.layer1 = EdgeGATv2(hidden, edge_dim=edge_dim, heads=heads, dropout=dropout)
        self.layer2 = EdgeGATv2(hidden, edge_dim=edge_dim, heads=heads, dropout=dropout)

    def forward(self, nodes, names, types, adj, edges, valid):
        h = self.node_geometry(nodes) + self.name_projection(names) + self.type_embedding(types)
        h = h * valid.unsqueeze(-1)
        h = self.layer1(h, edges, adj, valid)
        h = self.layer2(h, edges, adj, valid)
        return h
