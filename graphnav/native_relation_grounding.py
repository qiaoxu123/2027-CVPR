"""AerialVG-inspired *trainable* target-to-map-reference relation grounding.

This module is deliberately not the upstream AerialVG relation-transformer:
queries come from the frozen native encoder, while reference objects are
CityRefer named polygons/verified roads and relations use world (east,north)
coordinates. Only public map geometry and instruction-extracted relations
enter inference. Goal coordinates are used by loss functions exclusively.
"""
from __future__ import annotations
import math

import numpy as np
import torch
from torch import nn

from .semantic_geometry import (
    DIRECTIONS, RELATIONS, polygon_center, point_inside_polygon,
    prior_and_rule_fields,
)

RELATION_DIM = 8
MAX_RELATIONS = 8


def public_relation_descriptors(world_xy, graph, max_relations=MAX_RELATIONS,
                                coordinate_scale=400.):
    """Construct query/reference geometry from public instruction/map alone.

    Ambiguous same-name reference groups are excluded from this conservative
    pilot: selecting the GT-nearest name would leak the answer. Returns
    features [N,R,8], kinds [R], valid [R].
    """
    xy = np.asarray(world_xy, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError('Invalid candidate-world XY')
    if not 0 < max_relations <= MAX_RELATIONS or coordinate_scale <= 0:
        raise ValueError('Invalid relation capacity/coordinate scale')
    feats = np.zeros((len(xy), max_relations, RELATION_DIM), dtype=np.float32)
    types = np.zeros(max_relations, dtype=np.int64)
    valid = np.zeros(max_relations, dtype=np.bool_)
    entities = graph['entities']
    relations = graph['relations']
    if len(relations) > max_relations:
        raise ValueError('Relation count exceeds the public graph capacity')
    for j, rel in enumerate(relations):
        types[j] = RELATIONS.index(rel.kind)
        if not rel.anchors or any(len(group) != 1 for group in rel.anchors):
            # Duplicate landmark names are not silently disambiguated by GT.
            continue
        ids = [group[0] for group in rel.anchors]
        if not all(0 <= k < len(entities) for k in ids):
            raise ValueError('Invalid named map reference index')
        centers = [polygon_center(entities[k]['geometry_m']) for k in ids]
        center = np.mean(np.stack(centers, 0), axis=0)
        delta = xy - center[None]
        distance = np.linalg.norm(delta, axis=-1)
        _, _, rules = prior_and_rule_fields(
            xy, entities, [rel], sigma_m=25.,
        )
        direction = np.asarray(DIRECTIONS.get(rel.kind, (0., 0.)),
                               dtype=np.float32)
        length = float(np.linalg.norm(direction))
        if length:
            direction = direction / length
        # For road centerlines, "inside" is undefined and remains 0.
        e = entities[ids[0]]
        is_region = e['kind'] != 'road' and len(ids) == 1
        inside = (point_inside_polygon(xy, e['geometry_m']).astype(np.float32)
                  if is_region else np.zeros(len(xy), np.float32))
        feats[:, j, :] = np.stack([
            np.maximum(rules[:, 0], -50.) / 10.,
            delta[:, 0] / coordinate_scale,
            delta[:, 1] / coordinate_scale,
            np.minimum(distance / coordinate_scale, 5.),
            np.full(len(xy), direction[0], np.float32),
            np.full(len(xy), direction[1], np.float32),
            inside,
            np.clip((delta @ direction) / coordinate_scale, -5., 5.),
        ], axis=-1)
        valid[j] = True
    return (torch.from_numpy(feats), torch.from_numpy(types),
            torch.from_numpy(valid))


class RelationAwareQueryRanker(nn.Module):
    """Candidate-to-named-reference cross-attention with a bounded score head.

    On unmatched/no-relation instructions, identical native query scores and
    null are returned. The relation head starts at zero for a fair ablation.
    """

    def __init__(self, hidden=64, max_relations=MAX_RELATIONS, bound=3.):
        super().__init__()
        self.max_relations = max_relations
        self.bound = float(bound)
        self.candidate = nn.Sequential(
            nn.Linear(256 * 2 + 8, hidden), nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.reference = nn.Sequential(
            nn.Linear(RELATION_DIM, hidden), nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.relation_kind = nn.Embedding(len(RELATIONS), hidden)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.value = nn.Linear(hidden, hidden, bias=False)
        self.head = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        if not 0 < bound <= 10 or not 1 <= max_relations <= MAX_RELATIONS:
            raise ValueError('Invalid bounded relation head')

    def forward(self, features, text, text_mask, geometry, valid,
                relation_features, relation_types, relation_valid,
                native_logits, native_null, *, relation_enabled=True):
        b, n, dim = features.shape
        r = self.max_relations
        if (dim != 256 or text.shape[0] != b or text.shape[-1] != 256
                or text_mask.shape != text.shape[:2]
                or geometry.shape != (b, n, 8)
                or valid.shape != (b, n)
                or relation_features.shape != (b, n, r, RELATION_DIM)
                or relation_types.shape != (b, r)
                or relation_valid.shape != (b, r)
                or native_logits.shape != (b, n)
                or native_null.shape != (b,)):
            raise ValueError('Relation ranker public input shape mismatch')
        if not (torch.isfinite(features).all()
                and torch.isfinite(relation_features).all()
                and torch.isfinite(native_logits.masked_fill(~valid.bool(), 0)).all()):
            raise ValueError('Nonfinite public relation evidence')
        role_valid = relation_valid.bool() if relation_enabled else torch.zeros_like(
            relation_valid, dtype=torch.bool)
        has_relation = role_valid.any(-1)
        words = text_mask.float()
        sentence = (text.float() * words[..., None]).sum(1) / words.sum(
            1, keepdim=True).clamp_min(1)
        query = self.candidate(torch.cat([
            features.float(), sentence[:, None, :].expand(-1, n, -1),
            geometry.float(),
        ], -1))
        reference = (self.reference(relation_features.float())
                     + self.relation_kind(relation_types.long())[:, None])
        attn_logits = (self.query(query)[:, :, None, :]
                       * self.key(reference)).sum(-1) / math.sqrt(query.shape[-1])
        available = role_valid[:, None, :] & valid.bool()[:, :, None]
        attn = attn_logits.masked_fill(~available, -1e4).softmax(-1)
        attn = attn * available.float()
        attn = attn / attn.sum(-1, keepdim=True).clamp_min(1e-9)
        attended = (attn[..., None] * self.value(reference)).sum(2)
        delta = self.bound * torch.tanh(
            self.head(torch.cat([query, attended, query * attended], -1)).squeeze(-1))
        delta = torch.where(has_relation[:, None] & valid.bool(),
                            delta, torch.zeros_like(delta))
        logits = (native_logits.float() + delta).masked_fill(~valid.bool(), -1e4)
        return dict(
            logits=logits, null=native_null.float(),
            relation_delta=delta, attention=attn,
            has_relation=has_relation, relation_valid=role_valid,
        )


def weak_relation_ranking_loss(logits, valid, has_relation, query_world,
                               target_world, radius_m=20.):
    """Spatial-utility supervision only, not verified instance identity.

    No positives => skip entire row; do NOT invent a 'target absent' label.
    Rows without a uniquely matched named reference also contribute zero.
    """
    dist = (query_world.float() - target_world[:, None, :2].float()).norm(dim=-1)
    good = (dist <= radius_m) & valid.bool()
    bad = valid.bool() & ~good
    rows = good.any(-1) & bad.any(-1) & has_relation.bool()
    masked_logits = logits.float().masked_fill(~valid.bool(), -1e4)
    eligible = masked_logits.masked_fill(~good, -1e4)
    per_row = torch.logsumexp(masked_logits, -1) - torch.logsumexp(eligible, -1)
    loss = torch.where(rows, per_row, torch.zeros_like(per_row)).sum() / rows.sum().clamp_min(1)
    return loss, dict(
        supervised_rows=int(rows.sum().detach()),
        positive_rows=int(good.any(-1).sum().detach()),
        relation_rows=int(has_relation.sum().detach()),
    )
