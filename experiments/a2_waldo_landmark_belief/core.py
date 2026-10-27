"""A2 standalone, public-input-only landmark-conditioned box relation primitives.

This module does not call WALDO, change GraphNav's controller, or ingest private
GT. It is not a trained policy. Evidence from missing/ambiguous landmarks is
masked rather than interpreted as a negative observation.
"""
from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


PAIR_FEATURE_DIM = 12


def _polygon(poly: np.ndarray) -> np.ndarray | None:
    p = np.asarray(poly, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3:
        return None
    if not np.isfinite(p).all() or abs(cv2.contourArea(p)) <= 1e-6:
        return None
    return p


def _center(poly: np.ndarray) -> np.ndarray:
    moments = cv2.moments(poly)
    if abs(moments["m00"]) <= 1e-8:
        return poly.mean(axis=0)
    return np.array(
        [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
        dtype=np.float32,
    )


def box_landmark_geometry(
    box_polygons: Sequence[np.ndarray],
    landmark_polygons: Sequence[np.ndarray],
    box_confidences: Sequence[float],
    *,
    radius_m: float = 40.0,
    box_height_m: Sequence[float | None] | None = None,
    landmark_height_m: Sequence[float | None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Produce every N-by-L box/reference relationship in world metric XY.

    0 dx/r; 1 dy/r; 2 Euclidean distance/r; 3 sin(bearing); 4 cos(bearing);
    5 log(1+d)/log(1+r); 6 signed point-to-landmark-boundary distance/r;
    7 center inside landmark; 8 log(area box / area landmark)/5;
    9 detector box confidence; 10 observed height difference/20;
    11 height-valid mask. Positive x/y are east/north ONLY if the dataset
    coordinate transform has been independently checked to use that frame.

    The rectangular box projection is an approximation, NOT an object contour.
    Only PUBLIC detection polygons and PUBLIC linked landmark map polygons
    are accepted; never ground-truth target polygons or target coordinates.
    """
    if not (np.isfinite(radius_m) and radius_m > 0):
        raise ValueError("radius_m must be positive and finite")
    n, l = len(box_polygons), len(landmark_polygons)
    if len(box_confidences) != n:
        raise ValueError("box confidence count mismatch")
    if box_height_m is not None and len(box_height_m) != n:
        raise ValueError("box height count mismatch")
    if landmark_height_m is not None and len(landmark_height_m) != l:
        raise ValueError("landmark height count mismatch")

    out = np.zeros((n, l, PAIR_FEATURE_DIM), dtype=np.float32)
    valid = np.zeros((n, l), dtype=bool)
    boxes = [_polygon(p) for p in box_polygons]
    refs = [_polygon(p) for p in landmark_polygons]
    for i, box in enumerate(boxes):
        confidence = float(box_confidences[i])
        if box is None or not np.isfinite(confidence) or not 0 <= confidence <= 1:
            continue
        bx, by = _center(box)
        box_area = abs(cv2.contourArea(box))
        for j, ref in enumerate(refs):
            if ref is None:
                continue
            lx, ly = _center(ref)
            delta = np.array([bx - lx, by - ly], dtype=np.float64)
            distance = float(np.linalg.norm(delta))
            signed = float(cv2.pointPolygonTest(
                ref, (float(bx), float(by)), True,
            ))
            ref_area = abs(cv2.contourArea(ref))
            out[i, j] = [
                delta[0] / radius_m,
                delta[1] / radius_m,
                distance / radius_m,
                delta[1] / max(distance, 1e-6),
                delta[0] / max(distance, 1e-6),
                math.log1p(distance) / math.log1p(radius_m),
                signed / radius_m,
                float(signed >= 0),
                math.log(max(box_area, 1e-6) / max(ref_area, 1e-6)) / 5,
                confidence,
                0,
                0,
            ]
            if box_height_m is not None and landmark_height_m is not None:
                z_box, z_ref = box_height_m[i], landmark_height_m[j]
                if (z_box is not None and z_ref is not None and
                        np.isfinite(z_box) and np.isfinite(z_ref)):
                    out[i, j, 10] = (float(z_box) - float(z_ref)) / 20.0
                    out[i, j, 11] = 1.0
            valid[i, j] = True
    return out, valid


class InstructionLandmarkEncoder(nn.Module):
    """Per-landmark, contextual phrase + name + full-instruction encoding.

    BERT has already produced the frozen token representations (D=768 in the
    actual model). mention_mask MUST be aligned with BERT tokenizer offsets
    for a verified map-linked landmark; caller must reject ambiguous names.
    No token span means no positive landmark evidence, never an invented span.
    """

    def __init__(self, text_dim: int = 768, hidden: int = 128, heads: int = 4):
        super().__init__()
        self.phrase_name = nn.Sequential(
            nn.LayerNorm(text_dim * 2),
            nn.Linear(text_dim * 2, hidden),
            nn.GELU(),
        )
        self.full_sentence = nn.Linear(text_dim, hidden)
        self.context = nn.MultiheadAttention(
            hidden, heads, batch_first=True,
        )
        self.output = nn.LayerNorm(hidden)

    def forward(
        self,
        instruction_tokens: torch.Tensor,  # [B,T,D] frozen BERT
        token_valid: torch.Tensor,         # [B,T]
        mention_mask: torch.Tensor,        # [B,L,T], verified mention spans
        landmark_name_tokens: torch.Tensor, # [B,L,U,D] frozen BERT
        name_valid: torch.Tensor,          # [B,L,U]
        link_valid: torch.Tensor,          # [B,L], verified map linkage
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, d = instruction_tokens.shape
        if token_valid.shape != (b, t) or not token_valid.any(-1).all():
            raise ValueError("invalid full instruction token masks")
        if mention_mask.ndim != 3 or mention_mask.shape[:1] != (b,) or mention_mask.shape[-1] != t:
            raise ValueError("landmark mention spans must align with instruction tokens")
        l = mention_mask.shape[1]
        if (landmark_name_tokens.ndim != 4 or
                landmark_name_tokens.shape[:2] != (b, l) or
                landmark_name_tokens.shape[-1] != d or
                name_valid.shape != landmark_name_tokens.shape[:3] or
                link_valid.shape != (b, l)):
            raise ValueError("misaligned named landmark embeddings")
        if not all(torch.isfinite(v).all() for v in (instruction_tokens, landmark_name_tokens)):
            raise ValueError("nonfinite language encodings")
        mentions = mention_mask.bool() & token_valid[:, None, :].bool()
        names = name_valid.bool()
        eligible = link_valid.bool() & mentions.any(-1) & names.any(-1)
        phrase_w = mentions.to(instruction_tokens.dtype)
        name_w = names.to(landmark_name_tokens.dtype)
        phrase = torch.einsum("blt,btd->bld", phrase_w, instruction_tokens)
        phrase = phrase / phrase_w.sum(-1, keepdim=True).clamp_min(1)
        named = (landmark_name_tokens * name_w[..., None]).sum(-2)
        named = named / name_w.sum(-1, keepdim=True).clamp_min(1)
        query = self.phrase_name(torch.cat((phrase, named), -1))
        attended, _ = self.context(
            query, self.full_sentence(instruction_tokens),
            self.full_sentence(instruction_tokens),
            key_padding_mask=~token_valid.bool(),
            need_weights=False,
        )
        context = self.output(query + attended)
        return context * eligible[..., None], eligible


class BoxLandmarkRelationScorer(nn.Module):
    """Score boxes by all verified, instruction-conditioned landmark pairs."""

    def __init__(self, vision_dim: int = 1024, hidden: int = 128, pair_dim: int = PAIR_FEATURE_DIM):
        super().__init__()
        self.box = nn.Sequential(nn.LayerNorm(vision_dim), nn.Linear(vision_dim, hidden))
        self.geometry = nn.Sequential(nn.Linear(pair_dim, hidden), nn.GELU())
        self.pair_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.rank_head = nn.Sequential(
            nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden),
            nn.GELU(), nn.Linear(hidden, 1),
        )

    def forward(
        self,
        box_rgb: torch.Tensor,        # [B,N,D] frozen SigLIP2 vectors
        box_valid: torch.Tensor,      # [B,N] public boxes
        landmark_text: torch.Tensor,  # [B,L,H] per-mention encoded
        landmark_valid: torch.Tensor, # [B,L], verified linkage + span
        pair_geometry: torch.Tensor,  # [B,N,L,F]
        pair_valid: torch.Tensor,     # [B,N,L], geometry/linking support
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, n, _ = box_rgb.shape
        if landmark_text.ndim != 3 or landmark_text.shape[0] != b:
            raise ValueError("invalid landmark embeddings")
        l = landmark_text.shape[1]
        if (box_valid.shape != (b, n) or landmark_valid.shape != (b, l)
                or pair_valid.shape != (b, n, l)
                or pair_geometry.shape != (b, n, l, self.geometry[0].in_features)):
            raise ValueError("misaligned box-landmark pair tensors")
        if not torch.isfinite(box_rgb).all() or not torch.isfinite(pair_geometry).all():
            raise ValueError("nonfinite RGB or geometry")
        candidate = self.box(box_rgb.float())
        pair = torch.tanh(
            candidate[:, :, None, :] + landmark_text[:, None, :, :]
            + self.geometry(pair_geometry.float())
        )
        pair_scores = self.pair_head(pair).squeeze(-1)
        active = pair_valid.bool() & box_valid[:, :, None].bool() & landmark_valid[:, None, :].bool()
        # Softmax over verified landmarks only, even when a box has none.
        safe_logits = pair_scores.masked_fill(~active, -1e4)
        weights = safe_logits.softmax(-1) * active.to(safe_logits.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        pooled = (pair * weights[..., None]).sum(-2)
        rank = self.rank_head(torch.cat((candidate, pooled), -1)).squeeze(-1)
        reliable = active.any(-1)
        rank = rank.masked_fill(~reliable, -1e4)
        return rank, pair_scores.masked_fill(~active, -1e4), reliable


def conservative_belief_mix(
    a0_logits: torch.Tensor,            # [B,Q] actual dynamic A0 belief
    candidate_scores: torch.Tensor,     # [B,N]
    candidate_support: torch.Tensor,    # [B,N,Q] observed + local + projected
    candidate_eligible: torch.Tensor,   # [B,N] trustworthy matches
    alpha: float = 0.0,                 # calibrated externally; OFF by default
) -> torch.Tensor:
    """Reallocate only existing support mass; exact no-evidence fallback.

    Returns logits for the ORIGINAL Q-cell controller, with other cells'
    unnormalized logit values unchanged. The scorer itself is NOT calibration.
    """
    alpha = float(alpha)
    if not (math.isfinite(alpha) and 0 <= alpha <= .5):
        raise ValueError("alpha must be calibrated within [0, 0.5]")
    if alpha == 0:
        return a0_logits
    if (a0_logits.ndim != 2 or candidate_scores.ndim != 2
            or candidate_support.shape != (
                a0_logits.shape[0], candidate_scores.shape[1], a0_logits.shape[1],
            ) or candidate_eligible.shape != candidate_scores.shape):
        raise ValueError("belief grid and candidate support dimensions do not align")
    if not torch.isfinite(a0_logits).all() or not torch.isfinite(candidate_scores).all():
        raise ValueError("nonfinite belief or candidate scores")
    output = a0_logits.clone()
    for i in range(a0_logits.shape[0]):
        supports = candidate_support[i].bool()
        usable = candidate_eligible[i].bool() & supports.any(-1)
        if not usable.any():
            continue
        kernels = supports[usable].to(torch.float32)
        kernels = kernels / kernels.sum(-1, keepdim=True).clamp_min(1)
        weights = candidate_scores[i, usable].float().softmax(0)
        proposal = (weights[:, None] * kernels).sum(0)
        union = supports[usable].any(0)
        base = a0_logits[i].float()
        shift = base.max()
        raw = (base - shift).exp()
        mass = raw[union].sum()
        changed = (1 - alpha) * raw[union] + alpha * mass * proposal[union]
        output[i, union] = (shift + changed.clamp_min(1e-30).log()).to(output.dtype)
    return output
