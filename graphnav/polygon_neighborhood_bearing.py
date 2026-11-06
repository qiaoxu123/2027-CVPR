"""GT-free language + public Polygon neighborhood bearing predictor.

The model sees full BERT tokens, named reference embeddings, and geometry of
nearby PUBLIC polygons selected by reference distance only. A target ID, target
position, teacher trajectory, UAV yaw or RGB is never used in feature creation.
This is a direction-only diagnostic; it does not modify GeomBelief inference.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

POLYGON_TYPES = ("Other", "Building", "TrafficRoad", "Footpath", "Bridge",
                 "Parking", "Ground", "Car", "Water", "OtherVehicle")
TYPE_TO_ID = {name: i for i, name in enumerate(POLYGON_TYPES)}
NEIGHBOR_DIM = 8
ANCHOR_DIM = 5


def _describe_polygon(contour, radius):
    """Vertex-mean anchor matches the existing Oracle bearing convention."""
    xy = np.asarray(contour, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3 or not np.isfinite(xy).all():
        return None
    box = xy.max(0) - xy.min(0)
    area = 0.5 * abs(float(np.dot(xy[:, 0], np.roll(xy[:, 1], -1)) -
                            np.dot(xy[:, 1], np.roll(xy[:, 0], -1))))
    from shapely.geometry import Polygon
    if area <= 0 or not Polygon(xy).is_valid:
        return None
    # Major axis is UNSIGNED: doubled angle removes any spurious forward sign.
    centered = xy - xy.mean(0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    v = vh[0]
    theta = math.atan2(float(v[1]), float(v[0]))
    shape = np.array([box[0] / radius, box[1] / radius,
                      math.log1p(area) / math.log1p(radius * radius),
                      math.sin(2 * theta), math.cos(2 * theta)], dtype=np.float32)
    return xy.mean(0), shape


class PublicPolygonNeighborhood:
    """Deterministic nearest public polygons, NEVER nearest to the GT target."""

    def __init__(self, catalog, *, radius_m=150.0, max_neighbors=24, max_refs=16):
        if radius_m <= 0 or max_neighbors < 1 or max_refs < 1:
            raise ValueError("Invalid neighborhood capacity or radius")
        self.catalog = catalog
        self.radius_m = float(radius_m)
        self.max_neighbors = int(max_neighbors)
        self.max_refs = int(max_refs)
        self._map_cache = {}
        self._neighbor_cache = {}

    def _map_index(self, map_name):
        if map_name not in self._map_cache:
            if map_name not in self.catalog:
                raise KeyError("Public map not present: " + map_name)
            rows = {}
            for oid, obj in self.catalog[map_name].items():
                parsed = _describe_polygon(obj.get("contour", []), self.radius_m)
                if parsed is None:
                    continue
                center, shape = parsed
                rows[str(oid)] = (center, shape,
                                  TYPE_TO_ID.get(obj.get("object_type"), 0))
            self._map_cache[map_name] = rows
        return self._map_cache[map_name]

    def _neighbors(self, map_name, anchor_id):
        key = (map_name, str(anchor_id))
        if key in self._neighbor_cache:
            return self._neighbor_cache[key]
        rows = self._map_index(map_name)
        if str(anchor_id) not in rows:
            return None
        origin, shape, _ = rows[str(anchor_id)]
        options = []
        for oid, (center, geo, kind) in rows.items():
            if oid == str(anchor_id):
                continue
            delta = center - origin
            dist = float(np.linalg.norm(delta))
            if dist > self.radius_m:
                continue
            features = np.concatenate((
                np.array([delta[0] / self.radius_m,
                          delta[1] / self.radius_m,
                          dist / self.radius_m], dtype=np.float32),
                geo,
            ))
            options.append((dist, oid, features, kind))
        options.sort(key=lambda x: (x[0], x[1]))
        features = np.zeros((self.max_neighbors, NEIGHBOR_DIM), np.float32)
        types = np.zeros(self.max_neighbors, np.int64)
        mask = np.zeros(self.max_neighbors, bool)
        for j, (_, _, geo, kind) in enumerate(options[:self.max_neighbors]):
            features[j] = geo
            types[j] = kind
            mask[j] = True
        value = (origin.copy(), shape, features, types, mask)
        self._neighbor_cache[key] = value
        return value

    def for_references(self, map_name, references):
        """Public fields only; no GT or even a target ID argument exists."""
        if len(references) > self.max_refs:
            raise ValueError("Too many reference polygons; do not truncate silently")
        r, k = self.max_refs, self.max_neighbors
        anchors = np.zeros((r, ANCHOR_DIM), np.float32)
        neighbors = np.zeros((r, k, NEIGHBOR_DIM), np.float32)
        kinds = np.zeros((r, k), np.int64)
        valid = np.zeros((r, k), bool)
        reference_valid = np.zeros(r, bool)
        centers_world = np.zeros((r, 2), np.float64)
        for i, ref in enumerate(references):
            value = self._neighbors(map_name, str(ref["id"]))
            if value is None:
                continue
            center, shape, geo, kind, mask = value
            centers_world[i] = center
            anchors[i] = shape
            neighbors[i] = geo
            kinds[i] = kind
            valid[i] = mask
            reference_valid[i] = True
        return dict(anchor_geometry=anchors, neighbor_geometry=neighbors,
                    neighbor_types=kinds, neighbor_valid=valid,
                    landmark_valid=reference_valid,
                    anchor_centers_world=centers_world)


class NeighborhoodBearingHead(nn.Module):
    """Text-to-anchor attention followed by anchor-local Polygon aggregation.

    geometry_enabled=False is the matched text+name control with exactly
    the same parameters. Outputs world-map XY 8-bin probabilities as logits.
    """

    def __init__(self, *, text_dim=768, hidden=128, bins=8):
        super().__init__()
        if hidden < 8 or bins < 4 or bins % 2:
            raise ValueError("Require positive hidden and even angular bins")
        self.bins = int(bins)
        self.name = nn.Linear(text_dim, hidden)
        self.text_key = nn.Linear(text_dim, hidden)
        self.text_value = nn.Linear(text_dim, hidden)
        self.anchor = nn.Linear(ANCHOR_DIM, hidden, bias=False)
        self.neighbor = nn.Linear(NEIGHBOR_DIM, hidden, bias=False)
        self.neighbor_type = nn.Embedding(len(POLYGON_TYPES), hidden)
        self.local_query = nn.Linear(hidden * 2, hidden)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden * 3), nn.Linear(hidden * 3, hidden),
            nn.GELU(), nn.Linear(hidden, bins),
        )

    def forward(self, *, tokens, token_mask, node_emb, landmark_valid,
                anchor_geometry, neighbor_geometry, neighbor_types,
                neighbor_valid, geometry_enabled=True):
        if tokens.ndim != 3 or node_emb.ndim != 3:
            raise ValueError("Expected full BERT tokens and per-reference names")
        b, r, tdim = node_emb.shape
        if (tokens.shape[0] != b or tokens.shape[2] != tdim or
                token_mask.shape != tokens.shape[:2] or
                landmark_valid.shape != (b, r) or
                anchor_geometry.shape != (b, r, ANCHOR_DIM) or
                neighbor_geometry.shape[:2] != (b, r) or
                neighbor_geometry.shape[-1] != NEIGHBOR_DIM or
                neighbor_types.shape != neighbor_geometry.shape[:3] or
                neighbor_valid.shape != neighbor_types.shape):
            raise ValueError("Polygon/BERT alignment mismatch")
        if not token_mask.bool().any(-1).all():
            raise ValueError("Empty instruction")
        if not torch.isfinite(neighbor_geometry).all() or not torch.isfinite(anchor_geometry).all():
            raise ValueError("Nonfinite public geometry")
        if not geometry_enabled:
            anchor_geometry = torch.zeros_like(anchor_geometry)
            neighbor_valid = torch.zeros_like(neighbor_valid)
        name = self.name(node_emb.float())
        anchor = self.anchor(anchor_geometry.float())
        query = name + anchor
        keys = self.text_key(tokens.float())
        weights = torch.einsum("brh,blh->brl", query, keys) / math.sqrt(keys.shape[-1])
        weights = weights.masked_fill(~token_mask[:, None, :].bool(), -1e4).softmax(-1)
        text = torch.einsum("brl,blh->brh", weights, self.text_value(tokens.float()))
        geo = self.neighbor(neighbor_geometry.float()) + self.neighbor_type(neighbor_types)
        scorer = self.local_query(torch.cat((name, text), -1))
        local_score = (geo * scorer[:, :, None, :]).sum(-1) / math.sqrt(geo.shape[-1])
        local_score = local_score.masked_fill(~neighbor_valid.bool(), -1e4)
        local_weight = local_score.softmax(-1) * neighbor_valid.float()
        local_weight = local_weight / local_weight.sum(-1, keepdim=True).clamp_min(1e-8)
        local = (local_weight[..., None] * geo).sum(-2)
        logits = self.classifier(torch.cat((text, name + anchor, local), -1))
        return logits.masked_fill(~landmark_valid.bool()[..., None], 0.)
