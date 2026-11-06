"""Language-conditioned, explicit coarse-to-fine spatial relation graph.

Experimental TRAIN-only head; default UnifiedPGVB runtime is untouched.
Fine nodes: ALL 28x28 spatial cells (not oracle object proposals).
Coarse nodes: public named landmark polygons.
Intermediate nodes: public parking polygons selected without target labels.
Each edge carries explicit metric geometry, not just an attention embedding.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .candidate_geometry import FullCandidateDataset, relative_entity_features
from .semantic_geometry import canonical


def relation_mode(instruction: str) -> int:
    """0: unsupported, 1: direct coarse-to-fine, 2: car-in-parking-near-anchor.

    This intentionally conservative parser is NOT a complete relation parser.
    Unresolved left/right/behind are never mapped to geographic directions.
    """
    clean = canonical(instruction)
    parking = bool(re.search(r"\b(parking lot|parking area|car park|carpark)\b", clean))
    subject = re.sub(r"\bcar park\b", "", clean)
    vehicle = bool(re.search(r"\b(car|van|truck|vehicle|bus)\b", subject))
    inside = bool(re.search(r"\b(inside|within|in|parked in|parked at)\b", clean))
    if parking and vehicle and inside:
        return 2
    return 1


class RegionAugmentedCandidateDataset(FullCandidateDataset):
    """Add PUBLIC intermediate region polygons to an existing TRAIN record.

    The original FullCandidateDataset already yields every grid cell to anchor
    Polygon feature. This adapter adds cells->parking and parking->anchor edges.
    All region selection is language, public geometry and map identity only.
    TRAIN target_xy stays a label and is not read by the region builder.
    """
    def __init__(self, jsonl_path, *, objects_json, max_regions=12, **kwargs):
        super().__init__(jsonl_path, **kwargs)
        if max_regions < 1:
            raise ValueError("max_regions must be positive")
        self.max_regions = int(max_regions)
        self.objects = json.loads(Path(objects_json).read_text(encoding="utf-8"))
        self._region_cache = lru_cache(maxsize=128)(self._regions_for)

    def _regions_for(self, index):
        rec = self.records[index]
        anchors = self._prepared(index)[0]["entities"]
        mode = relation_mode(rec["instruction"])
        q = len(self.grid_xy)
        n = self.capacity
        m = self.max_regions
        qr = np.zeros((q, m, 10), dtype=np.float16)
        ra = np.zeros((m, n, 10), dtype=np.float16)
        valid = np.zeros(m, dtype=bool)
        kinds = np.zeros(m, dtype=np.int64)
        # No named anchor or ungrounded relation: don't invent intermediate nodes.
        if mode != 2 or not anchors:
            return qr, ra, valid, kinds, mode
        bounds = np.asarray(rec["world_bounds"], dtype=np.float32)
        low = bounds[:2]
        span = bounds[2:] - low
        if (span <= 0).any():
            raise ValueError("invalid public world bounds")
        anchor_boxes = [
            (np.asarray(e["geometry_m"]).min(0), np.asarray(e["geometry_m"]).max(0))
            for e in anchors
        ]
        eligible = []
        area = self.objects.get(rec["map_name"], {})
        for object_id, obj in area.items():
            if str(obj.get("object_type", "")).casefold() != "parking":
                continue
            poly = np.asarray(obj.get("contour", []), dtype=np.float32)
            if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3:
                continue
            if not np.isfinite(poly).all():
                continue
            center = poly.mean(0)
            # Exact lower bound to anchor AABBs. Never use GT to select regions.
            rank = min(
                float(np.linalg.norm(np.maximum(np.maximum(lo-center, center-hi), 0)))
                for lo, hi in anchor_boxes
            )
            eligible.append((rank, str(object_id), poly, center))
        eligible.sort(key=lambda z: (z[0], z[1]))
        chosen = eligible[:m]
        if chosen:
            regions = [{"geometry_m": p, "kind": "parking"}
                       for _, _, p, _ in chosen]
            world_grid = low[None] + self.grid_xy * span[None]
            qr[:, :len(regions)] = relative_entity_features(
                world_grid, regions, len(regions)
            ).astype(np.float16)
            centers = np.stack([c for _, _, _, c in chosen], axis=0)
            ra[:len(regions)] = relative_entity_features(
                centers, anchors, n
            ).astype(np.float16)
            valid[:len(regions)] = True
            kinds[:len(regions)] = 1  # Parking (explicit semantic type)
        return qr, ra, valid, kinds, mode

    def __getitem__(self, index):
        row = super().__getitem__(index)
        qr, ra, valid, kinds, mode = self._region_cache(index)
        row["region_grid_geometry"] = torch.from_numpy(qr.copy())
        row["region_anchor_geometry"] = torch.from_numpy(ra.copy())
        row["region_valid"] = torch.from_numpy(valid.copy())
        row["region_types"] = torch.from_numpy(kinds.copy())
        row["relation_mode"] = mode
        return row


def _masked_softmax(scores, mask, dim=-1):
    mask = mask.bool()
    masked = scores.float().masked_fill(~mask, -1e4)
    weights = torch.softmax(masked, dim=dim) * mask.to(scores.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-8)


class HierarchicalRelationGraph(nn.Module):
    """Anchor -> intermediate region -> dense target-node attention.

    The model learns how to combine geometry and language. A geometric
    log-likelihood is also explicitly computed on every path. All paths can
    abstain; a missing intermediary never becomes a fabricated one.
    """
    def __init__(self, context_dim=128, text_dim=768, hidden=96):
        super().__init__()
        self.query = nn.Sequential(nn.Linear(context_dim, hidden), nn.GELU())
        self.lang = nn.Sequential(nn.Linear(text_dim, hidden), nn.LayerNorm(hidden))
        self.anchor = nn.Sequential(nn.Linear(text_dim, hidden), nn.LayerNorm(hidden))
        self.kind = nn.Embedding(2, hidden)
        self.region_seed = nn.Sequential(nn.Linear(10, hidden), nn.GELU())
        self.edge_qa = nn.Linear(10, hidden)
        self.edge_qr = nn.Linear(10, hidden)
        self.edge_ra = nn.Linear(10, hidden)
        self.score_ra = nn.Linear(hidden, 1)
        self.score_qa = nn.Linear(hidden, 1)
        self.score_qr = nn.Linear(hidden, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 4),
            nn.Linear(hidden * 4, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.learned_scale = nn.Parameter(torch.tensor(1.0))
        self.geometric_scale = nn.Parameter(torch.tensor(0.25))

    def forward(
        self, *, base_logits, candidate_context, tokens, token_mask,
        anchor_embeddings, anchor_valid, pair_geometry,
        region_grid_geometry, region_anchor_geometry, region_valid,
        region_types, relation_mode, ablation="full",
    ):
        if ablation not in ("full", "flat", "no-geometry", "no-language"):
            raise ValueError("unknown ablation")
        b, q, h0 = candidate_context.shape
        b2, q2, a, d = pair_geometry.shape
        r = region_valid.shape[1]
        if (b2, q2, d) != (b, q, 10) or base_logits.shape != (b, q):
            raise ValueError("invalid candidate or geometric feature shapes")
        if anchor_embeddings.shape != (b, a, tokens.shape[-1]):
            raise ValueError("anchor/token semantic feature mismatch")
        if anchor_valid.shape != (b, a) or token_mask.shape != tokens.shape[:2]:
            raise ValueError("invalid anchor/token masks")
        if (region_grid_geometry.shape != (b, q, r, 10)
                or region_anchor_geometry.shape != (b, r, a, 10)
                or region_types.shape != (b, r)
                or relation_mode.shape != (b,)):
            raise ValueError("invalid intermediate region features")
        if not token_mask.any(dim=1).all():
            raise ValueError("each query needs language tokens")
        if not all(torch.isfinite(x).all() for x in (
            base_logits, candidate_context, tokens, anchor_embeddings,
            pair_geometry, region_grid_geometry, region_anchor_geometry,
        )):
            raise ValueError("nonfinite explicit graph input")

        use_geom = ablation != "no-geometry"
        mode = relation_mode.long()
        anchor_any = anchor_valid.any(-1)
        indirect = ((mode == 2) & region_valid.any(-1) & anchor_any
                    & (ablation not in ("flat",)))
        direct = (mode == 1) & anchor_any
        active = direct | indirect
        text_mean = ((tokens * token_mask[..., None].float()).sum(1) /
                     token_mask.sum(1, keepdim=True).clamp_min(1))
        lang = self.lang(text_mean)
        if ablation == "no-language":
            lang = lang * 0.
        query = self.query(candidate_context)
        anchors = self.anchor(anchor_embeddings)
        ga = self.edge_qa(pair_geometry.float()) if use_geom else torch.zeros(
            b, q, a, query.shape[-1], device=query.device, dtype=query.dtype)
        gr = self.edge_qr(region_grid_geometry.float()) if use_geom else torch.zeros(
            b, q, r, query.shape[-1], device=query.device, dtype=query.dtype)
        gra = self.edge_ra(region_anchor_geometry.float()) if use_geom else torch.zeros(
            b, r, a, query.shape[-1], device=query.device, dtype=query.dtype)

        # Direct: target grid node attends to all referenced Polygon nodes.
        da = self.score_qa(torch.tanh(
            query[:, :, None] + anchors[:, None] + ga +
            lang[:, None, None]
        )).squeeze(-1)
        w_qa = _masked_softmax(da, anchor_valid[:, None, :].expand(-1, q, -1))
        msg_direct = torch.einsum("bqa,bah->bqh", w_qa, anchors)

        # First pass: an intermediate public parking polygon attends to
        # named landmarks; second pass: each fine cell attends to intermediates.
        region_features = self.kind(region_types.long()) + self.region_seed(
            region_anchor_geometry.float().mean(2)
        )
        sa = self.score_ra(torch.tanh(
            region_features[:, :, None] + anchors[:, None] + gra +
            lang[:, None, None]
        )).squeeze(-1)
        w_ra = _masked_softmax(
            sa, anchor_valid[:, None, :].expand(-1, r, -1))
        grounded_regions = region_features + torch.einsum(
            "bra,bah->brh", w_ra, anchors)
        sr = self.score_qr(torch.tanh(
            query[:, :, None] + grounded_regions[:, None] + gr +
            lang[:, None, None]
        )).squeeze(-1)
        w_qr = _masked_softmax(
            sr, region_valid[:, None, :].expand(-1, q, -1))
        msg_indirect = torch.einsum(
            "bqr,brh->bqh", w_qr, grounded_regions)

        raw = self.head(torch.cat((
            query, msg_direct, msg_indirect, lang[:, None].expand(-1, q, -1)
        ), -1)).squeeze(-1)
        raw = raw - raw.mean(1, keepdim=True)

        # Signed boundary distance is in physical meters / 400. Containment
        # is an explicit Boolean at index 6, never an inferred "attention".
        if use_geom:
            direct_near = -torch.relu(pair_geometry[..., 5].float()) * 400. / 35.
            direct_score = (w_qa * direct_near.clamp(-8, 0)).sum(-1)
            inner = (2.0 * region_grid_geometry[..., 6].float()
                     - torch.relu(region_grid_geometry[..., 5].float()) * 400. / 25.)
            near = (-torch.relu(region_anchor_geometry[..., 5].float())
                    * 400. / 50.).clamp(-8, 0)
            near_region = (w_ra * near).sum(-1)
            indirect_score = (w_qr * (inner + near_region[:, None])).sum(-1)
            explicit = torch.where(indirect[:, None], indirect_score, direct_score)
        else:
            explicit = torch.zeros_like(raw)
        # A missing public anchor/intermediate produces EXACT base logits.
        correction = (
            torch.tanh(self.learned_scale) * raw.clamp(-8, 8)
            + torch.tanh(self.geometric_scale) * explicit.clamp(-8, 8)
        )
        refined = base_logits.float() + torch.where(
            active[:, None], correction, torch.zeros_like(correction))
        return {
            "logits": refined,
            "probabilities": refined.softmax(-1),
            "delta": refined - base_logits.float(),
            "direct_active": direct,
            "indirect_active": indirect,
            "unresolved": ~active,
            "anchor_attention": w_qa,
            "region_attention": w_qr,
            "region_anchor_attention": w_ra,
        }
