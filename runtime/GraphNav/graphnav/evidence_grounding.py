"""Opt-in, target-conditioned evidence refinement on top of a frozen A3 GoalField.

Only observed camera patches, public map coordinates and instruction tokens enter
forward(). Labels / future observations are strictly loss-side. A3 checkpoints
and existing DynamicGoalBelief checkpoint schemas remain unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re

import torch
from torch import nn
from torch.nn import functional as F

from .full_localization import GoalField


_APPEARANCE = re.compile(
    r"\b(?:black|white|grey|gray|brown|red|blue|green|yellow|orange|"
    r"beige|silver|brick|roof|rooftop|house|building|car|vehicle|"
    r"parking|chimney|skylight|garage|trampoline|rectangular|l.shaped)\b",
    re.IGNORECASE,
)


def appearance_query(instruction: str) -> str:
    """Conservative visual query; keep context when no attributes are detected.

    This is a query builder, NOT a semantic parse or a source of supervision.
    Map names remain in the full BERT geometric branch.
    """
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("nonempty instruction required")
    # Capture appearance expressions with context; do not claim that these
    # phrases or unobserved targets are automatically grounded.
    words = instruction.split()
    kept = set()
    for i, word in enumerate(words):
        if _APPEARANCE.search(word):
            kept.update(range(max(0, i - 2), min(len(words), i + 3)))
    return " ".join(words[i] for i in sorted(kept)) if kept else instruction


@dataclass
class RasterizedEvidence:
    scores: torch.Tensor  # [B, Q] matching evidence; undefined cells are exactly zero
    coverage: torch.Tensor  # [B, Q] in [0,1], zero = unknown


def rasterize_observed_patches(
    patch_scores: torch.Tensor,
    patch_xy: torch.Tensor,
    patch_valid: torch.Tensor,
    grid_xy: torch.Tensor,
    *,
    sigma: float = 0.035,
    cutoff_sigma: float = 2.5,
) -> RasterizedEvidence:
    """Spread evidence within the actually observed patch support.

    All coordinates use the SAME public map-normalized XY frame. A camera
    footprint and source validity mask must be used by the caller to construct
    patch_xy/patch_valid; this function never synthesizes unobserved pixels.
    """
    if sigma <= 0 or cutoff_sigma <= 0:
        raise ValueError("positive raster radius required")
    if patch_scores.ndim != 2 or patch_xy.shape != (*patch_scores.shape, 2):
        raise ValueError("patch score/XY mismatch")
    if grid_xy.ndim != 3 or grid_xy.shape[0] != patch_scores.shape[0] or grid_xy.shape[-1] != 2:
        raise ValueError("grid dimensions mismatch")
    if patch_valid.shape != patch_scores.shape or patch_valid.dtype != torch.bool:
        raise ValueError("valid observed-patch boolean mask required")
    if not all(torch.isfinite(t).all() for t in (patch_scores, patch_xy, grid_xy)):
        raise ValueError("nonfinite patch evidence or geometry")
    d = torch.cdist(grid_xy.float(), patch_xy.float())
    support = (d <= sigma * cutoff_sigma) & patch_valid[:, None, :]
    kernel = torch.exp(-0.5 * (d / sigma).square()) * support
    mass = kernel.sum(dim=-1)
    value = (kernel * patch_scores.float()[:, None, :]).sum(dim=-1) / mass.clamp_min(1e-12)
    # Normalized support is a confidence, not proof of target presence/absence.
    coverage = (mass / 2.0).clamp(0.0, 1.0)
    return RasterizedEvidence(torch.where(mass > 0, value, torch.zeros_like(value)), coverage)


class SpatialEvidenceMemory:
    """Episode-local causal mean with viewpoint de-duplication.

    An identical/collocated view is not repeated as independent evidence.
    History stores detached evidence; it is not a differentiable recurrent model.
    """

    def __init__(self, batch: int, cells: int, device, *, min_move_m=2.0, min_turn_rad=0.26):
        self.sums = torch.zeros(batch, cells, device=device, dtype=torch.float32)
        self.weights = torch.zeros_like(self.sums)
        self.last_step = -1
        self.last_pose = None
        self.min_move_m = float(min_move_m)
        self.min_turn_rad = float(min_turn_rad)

    def update(self, evidence: RasterizedEvidence, *, step: int, pose_xyyaw: torch.Tensor):
        if step <= self.last_step:
            raise ValueError("visual memory steps must advance strictly")
        if (pose_xyyaw.ndim != 2 or pose_xyyaw.shape[0] != self.sums.shape[0]
                or pose_xyyaw.shape[1] not in (3, 4) or not torch.isfinite(pose_xyyaw).all()):
            raise ValueError("pose must be finite [B,3] x/y/yaw or [B,4] x/y/z/yaw")
        if evidence.scores.shape != self.sums.shape or evidence.coverage.shape != self.sums.shape:
            raise ValueError("evidence grid changed within episode")
        if not all(torch.isfinite(t).all() for t in (evidence.scores, evidence.coverage)):
            raise ValueError("nonfinite evidence")
        if ((evidence.coverage < 0) | (evidence.coverage > 1)).any():
            raise ValueError("coverage outside [0,1]")
        accepts = torch.ones(self.sums.shape[0], device=self.sums.device, dtype=torch.bool)
        pose = pose_xyyaw.detach().float().to(self.sums.device)
        if self.last_pose is not None:
            if pose.shape != self.last_pose.shape:
                raise ValueError("pose schema changed within episode")
            # A genuine height change changes sampling, even at fixed XY.
            # Nadir yaw-only rotation does not add an independent physical view.
            dimensions = 3 if pose.shape[1] == 4 else 2
            distance = torch.linalg.vector_norm(
                pose[:, :dimensions] - self.last_pose[:, :dimensions], dim=-1
            )
            accepts = distance >= self.min_move_m
        w = evidence.coverage.detach().float().to(self.sums.device) * accepts[:, None]
        self.sums += evidence.scores.detach().float().to(self.sums.device) * w
        self.weights += w
        self.last_step = int(step)
        # Keep last *accepted* viewpoint. A duplicate frame must not prevent
        # a later slowly accumulated significant movement from being accepted.
        self.last_pose = pose if self.last_pose is None else torch.where(
            accepts[:, None], pose, self.last_pose
        )
        return self.snapshot()

    def snapshot(self) -> RasterizedEvidence:
        known = self.weights > 0
        values = self.sums / self.weights.clamp_min(1e-8)
        return RasterizedEvidence(
            torch.where(known, values, torch.zeros_like(values)),
            self.weights.clamp(0, 1),
        )


class TargetEvidenceRefiner(nn.Module):
    """Independent trainable adapter; has no impact until explicitly loaded.

    relation_fields are existing public A3 per-cell geometry/rule features.
    The visual path compares appearance-query BERT tokens with frozen actual
    observed SigLIP2 patch tokens using *trained* cross-modal projections.
    No pretrained BERT<->SigLIP alignment is assumed.
    """

    def __init__(self, *, hidden=128, text_dim=768, vision_dim=1024, relation_dim=8,
                 sigma=0.035, max_delta=3.0):
        super().__init__()
        self.hidden, self.sigma, self.max_delta = int(hidden), float(sigma), float(max_delta)
        self.relation = nn.Sequential(
            nn.LayerNorm(hidden + relation_dim),
            nn.Linear(hidden + relation_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.text_projection = nn.Sequential(nn.Linear(text_dim, hidden), nn.LayerNorm(hidden))
        self.patch_projection = nn.Sequential(nn.Linear(vision_dim, hidden), nn.LayerNorm(hidden))
        self.match_scale = nn.Parameter(torch.tensor(1.0))
        # Both zero: exact old A3/DynamicGoalBelief behavior at initialization.
        self.relation_gate = nn.Parameter(torch.zeros(()))
        self.visual_gate = nn.Parameter(torch.zeros(()))

    def appearance_scores(self, text_tokens, text_mask, patch_features, patch_valid):
        if text_tokens.ndim != 3 or text_mask.shape != text_tokens.shape[:2]:
            raise ValueError("appearance text tokens/mask mismatch")
        if patch_features.ndim != 3 or patch_valid.shape != patch_features.shape[:2]:
            raise ValueError("observed patch tokens/mask mismatch")
        if patch_features.shape[0] != text_tokens.shape[0]:
            raise ValueError("appearance/patch batch mismatch")
        if not text_mask.bool().any(-1).all():
            raise ValueError("each appearance query requires at least one valid token")
        if not all(torch.isfinite(t).all() for t in (text_tokens, patch_features)):
            raise ValueError("nonfinite appearance embeddings")
        language = self.text_projection(text_tokens.float())
        mask = text_mask.bool()
        pooled = (language * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        query = F.normalize(pooled, dim=-1)
        patch = F.normalize(self.patch_projection(patch_features.float()), dim=-1)
        scores = (patch * query[:, None, :]).sum(-1) * self.match_scale.exp().clamp(max=20)
        return torch.where(patch_valid.bool(), scores, torch.zeros_like(scores))

    def forward(self, field: GoalField, *, grid_xy, relation_fields=None,
                text_tokens=None, text_mask=None, patch_features=None,
                patch_xy=None, patch_valid=None, memory=None, step=None, pose_xyyaw=None,
                observed_grid_valid=None):
        if field.dense_context is None:
            raise ValueError("refiner needs original A3 dense candidate features")
        logits = field.belief_logits
        b, q = logits.shape
        if grid_xy.shape != (b, q, 2):
            raise ValueError("full-map grid must match A3 field")
        if relation_fields is not None:
            if relation_fields.shape[:2] != (b, q):
                raise ValueError("relation map is not aligned with all candidates")
            nonempty = relation_fields.detach().abs().sum(-1) > 0
            relation_input = torch.cat((field.dense_context.float(), relation_fields.float()), -1)
            delta = self.relation(relation_input).squeeze(-1)
            logits = logits + torch.tanh(self.relation_gate) * delta * nonempty
        if patch_features is not None:
            if any(x is None for x in (text_tokens, text_mask, patch_xy, patch_valid)):
                raise ValueError("visual branch needs actual patches, positions and query")
            raw_scores = self.appearance_scores(
                text_tokens, text_mask, patch_features, patch_valid
            )
            image_evidence = rasterize_observed_patches(
                raw_scores, patch_xy, patch_valid, grid_xy, sigma=self.sigma
            )
            if observed_grid_valid is not None:
                if (observed_grid_valid.shape != logits.shape
                        or observed_grid_valid.dtype != torch.bool):
                    raise ValueError("observed grid support must be aligned boolean [B,Q]")
                image_evidence = RasterizedEvidence(
                    image_evidence.scores * observed_grid_valid,
                    image_evidence.coverage * observed_grid_valid,
                )
            if memory is not None:
                if step is None or pose_xyyaw is None:
                    raise ValueError("causal memory needs step and public metric pose")
                image_evidence = memory.update(
                    image_evidence, step=step, pose_xyyaw=pose_xyyaw
                )
            visual_residual = image_evidence.scores.clamp(-self.max_delta, self.max_delta)
            logits = logits + torch.tanh(self.visual_gate) * (
                visual_residual * image_evidence.coverage
            )
        elif memory is not None and memory.weights.any():
            # Previously seen evidence remains available during a temporarily
            # invalid current frame; unknown cells are never penalized.
            history = memory.snapshot()
            logits = logits + torch.tanh(self.visual_gate) * (
                history.scores.clamp(-self.max_delta, self.max_delta) * history.coverage
            )
        k = field.candidate_ids.shape[1]
        ids = logits.topk(k, dim=-1).indices
        return GoalField(
            belief_logits=logits, candidate_ids=ids,
            candidate_xy=grid_xy.gather(1, ids[..., None].expand(-1, -1, 2)),
            candidate_context=field.dense_context.gather(
                1, ids[..., None].expand(-1, -1, self.hidden)
            ), visual_coverage=(
                memory.snapshot().coverage if memory is not None else None
            ), dense_context=field.dense_context,
        )
