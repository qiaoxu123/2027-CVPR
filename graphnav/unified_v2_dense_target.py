"""TRAIN-label-only polygon supervision and efficient 384² global Belief.

The CityRefer target Polygon and GT RGB crop are *private training labels*.
They never enter UnifiedPGVB.forward_field(), forward_dense_field(), or runtime.
384² is a spatial distribution, NOT 147456 expensive graph-attention nodes.
Coordinates: tensor row increases with WORLD +Y (same as grid_centers).
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def polygon_mask_world(vertices_world, world_bounds, resolution=384):
    """Rasterize one supervised world-XY Polygon to map-aligned [H,W] bool.

    Returns (mask,valid); invalid, wholly off-map, or subpixel polygons are
    *unavailable* supervision, not fake point-centered GT masks.
    """
    if resolution < 16:
        raise ValueError("resolution must be >=16")
    v = np.asarray(vertices_world, dtype=np.float64)
    b = np.asarray(world_bounds, dtype=np.float64)
    out = np.zeros((resolution, resolution), dtype=np.float32)
    if (b.shape != (4,) or not np.isfinite(b).all()
            or np.any(b[2:] <= b[:2])):
        raise ValueError("invalid public map bounds")
    if (v.ndim != 2 or v.shape[1] != 2 or len(v) < 3
            or not np.isfinite(v).all()):
        return out, False
    # Never silently clip malformed, self-intersecting polygons into labels.
    try:
        from shapely.geometry import Polygon
        poly = Polygon(v)
        if not poly.is_valid or poly.area <= 0:
            return out, False
    except ImportError:
        edge = np.roll(v, -1, axis=0)
        if abs(float((v[:, 0] * edge[:, 1] - edge[:, 0] * v[:, 1]).sum())) < 1e-7:
            return out, False
    normalized = (v - b[:2]) / (b[2:] - b[:2])
    if (normalized.max(0) < 0).any() or (normalized.min(0) > 1).any():
        return out, False
    # 2x coverage sampling keeps small targets from arbitrary binary snapping.
    aa = 2
    scale = aa * resolution
    coords = np.rint(normalized * scale - .5).astype(np.int32)
    high = np.zeros((scale, scale), dtype=np.uint8)
    cv2.fillPoly(high, [coords], 1)
    out = cv2.resize(high.astype(np.float32), (resolution, resolution),
                     interpolation=cv2.INTER_AREA)
    return out, bool(out.sum() > 0)


def polygon_supervision_batch(polygons_world, world_bounds, resolution=384, device=None):
    if len(polygons_world) != len(world_bounds):
        raise ValueError("Polygon and map bound batches disagree")
    masks, valid = [], []
    for poly, bounds in zip(polygons_world, world_bounds):
        mask, ok = polygon_mask_world(poly, bounds, resolution)
        masks.append(mask)
        valid.append(ok)
    return (torch.from_numpy(np.stack(masks)).to(device),
            torch.tensor(valid, dtype=torch.bool, device=device))


def polygon_belief_loss(dense_logits, target_masks, valid, *, blur=3):
    """Weak region-mass likelihood; NOT BCE/Dice pixel-instance segmentation.

    The CityRefer object contour may be coarser than the physical object.
    We reward probability mass within a softly dilated region, rather than
    penalizing every pixel outside an unverified instance mask as 'negative'.
    """
    if (dense_logits.ndim != 3 or target_masks.shape != dense_logits.shape
            or valid.shape != dense_logits.shape[:1] or valid.dtype != torch.bool
            or blur < 0 or not math.isfinite(float(blur))):
        raise ValueError("invalid dense polygon supervision")
    if not torch.isfinite(dense_logits).all() or not torch.isfinite(target_masks).all():
        raise ValueError("nonfinite dense supervision")
    if ((target_masks < 0) | (target_masks > 1)).any():
        raise ValueError("mask coverage outside [0,1]")
    if not valid.any():
        return dense_logits.sum() * 0.
    weights = target_masks.float().unsqueeze(1)
    if blur:
        kernel = 2 * int(blur) + 1
        weights = F.max_pool2d(weights, kernel, stride=1, padding=int(blur))
    probs = dense_logits.float().flatten(1).softmax(-1)
    mass = (probs * weights.flatten(1)).sum(-1).clamp_min(1e-9)
    return -mass[valid].log().mean()


def _observed_evidence_canvas(evidence, size, *, device, dtype):
    """Differentiable sparse observation scores from the *existing* PGVB."""
    if evidence is None:
        return None
    uv, scores, valid = evidence
    if (uv.ndim != 3 or uv.shape[-1] != 2 or scores.shape != uv.shape[:2]
            or valid.shape != scores.shape or valid.dtype != torch.bool):
        raise ValueError("invalid observation sparse evidence")
    b, n = scores.shape
    good = valid & torch.isfinite(uv).all(-1)
    if not good.any():
        return torch.zeros(b, 2, size, size, device=device, dtype=dtype)
    coords = (uv.float() * size).floor().long().clamp(0, size - 1)
    good &= (uv >= 0).all(-1) & (uv <= 1).all(-1)
    flat = coords[..., 1] * size + coords[..., 0]
    w = good.float()
    counts = scores.new_zeros(b, size * size, dtype=torch.float32).scatter_add(
        1, flat, w)
    values = scores.new_zeros(b, size * size, dtype=torch.float32).scatter_add(
        1, flat, scores.float() * w)
    avg = values / counts.clamp_min(1.)
    covered = (counts > 0).float()
    # Spread sparse current-view patch evidence to adjacent map pixels.
    evidence_map = torch.stack((avg, covered), dim=1).reshape(b, 2, size, size)
    return F.avg_pool2d(evidence_map, 9, stride=1, padding=4)


class DenseBeliefDecoder(nn.Module):
    """Lightweight learned *spatial decoder*, NOT another vision encoder.

    Coarse 28² graph context remains the global reasoning trunk. The dense map
    receives the SAME causal PGVB evidence. It cannot recover unseen optical
    detail purely by interpolating coarse logits; report this limitation.
    """
    def __init__(self, coarse_size=28, output_size=384, hidden=16):
        super().__init__()
        if coarse_size < 2 or output_size < coarse_size:
            raise ValueError("invalid coarse/fine Belief sizes")
        self.coarse_size = int(coarse_size)
        self.output_size = int(output_size)
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.gain = nn.Parameter(torch.tensor(0.5))

    def forward(self, coarse_logits, evidence=None):
        b, q = coarse_logits.shape
        if q != self.coarse_size ** 2 or not torch.isfinite(coarse_logits).all():
            raise ValueError("expected finite coarse grid logits")
        size = self.output_size
        coarse = coarse_logits.float().reshape(b, 1, self.coarse_size, self.coarse_size)
        # Logit-centered interpolation prevents arbitrary global logit offsets
        # from making the refiner depend on a meaningless constant.
        coarse = coarse - coarse.mean(dim=(-2, -1), keepdim=True)
        up = F.interpolate(coarse, (size, size), mode="bilinear", align_corners=False)
        canvas = _observed_evidence_canvas(
            evidence, size, device=coarse_logits.device, dtype=coarse_logits.dtype)
        if canvas is None:
            canvas = up.new_zeros(b, 2, size, size)
        residual = self.net(torch.cat((up, canvas), dim=1)).squeeze(1)
        return up.squeeze(1) + torch.tanh(self.gain) * torch.tanh(residual)


def highres_to_controller(dense_logits, coarse_size=28):
    """Marginalize 384² *probability mass* to the 28² controller interface."""
    if dense_logits.ndim != 3 or dense_logits.shape[-1] != dense_logits.shape[-2]:
        raise ValueError("expected [B,H,W] global Belief")
    probs = dense_logits.float().flatten(1).softmax(-1).reshape_as(dense_logits)
    low = F.interpolate(probs.unsqueeze(1), size=(coarse_size, coarse_size),
                        mode="area").flatten(1)
    low = low / low.sum(-1, keepdim=True).clamp_min(1e-12)
    return low.clamp_min(1e-12).log()


def target_visual_alignment_loss(
    observed_features, observed_valid, observed_uv,
    teacher_features, polygons_world, world_bounds, visual_projection,
    *, temperature=.1, min_positives=1,
):
    """InfoNCE-style multi-positive target appearance supervision.

    Teacher image vectors originate from GT-only RGB crops and MUST NOT be
    supplied to model.forward or used for proposal generation. Positive
    observed patches must overlap target Polygon. No observed positive means
    EXACT ZERO visual loss, rather than an oracle target-centered observation.

    This is a geometric visibility proxy; not verified instance segmentation.
    """
    if not 0 < temperature <= 1:
        raise ValueError("invalid temperature")
    b, n, scales, d = observed_features.shape
    if (observed_uv.shape != (b, n, 2)
            or observed_valid.shape != (b, n)
            or teacher_features.shape != (b, d)
            or len(polygons_world) != b or len(world_bounds) != b):
        raise ValueError("visual supervision input shapes disagree")
    if observed_valid.dtype != torch.bool:
        raise ValueError("observed_valid must be bool")
    # A masked native pool per candidate is already produced by ONE frozen
    # SigLIP2 forward per actual observed frame.
    observed = observed_features.float().mean(2)
    pos = np.zeros((b, n), dtype=bool)
    for j, (polygon, bounds) in enumerate(zip(polygons_world, world_bounds)):
        xy = observed_uv[j].detach().cpu().numpy()
        world = np.asarray(bounds)[:2] + xy * (
            np.asarray(bounds)[2:] - np.asarray(bounds)[:2])
        try:
            from .semantic_geometry import point_inside_polygon
            if polygon is not None and len(polygon) >= 3:
                pos[j] = point_inside_polygon(world, polygon)
        except (ValueError, TypeError):
            pass
    positive = torch.from_numpy(pos).to(device=observed.device) & observed_valid
    eligible = ((positive.sum(-1) >= min_positives)
                & (observed_valid & ~positive).any(-1))
    if not eligible.any():
        return observed_features.float().sum() * 0., eligible
    teacher = F.normalize(visual_projection(teacher_features.float()), dim=-1)
    student = F.normalize(visual_projection(observed), dim=-1)
    logits = (student * teacher[:, None]).sum(-1) / temperature
    all_logits = torch.logsumexp(logits.masked_fill(~observed_valid, -1e4), -1)
    positive_logits = torch.logsumexp(logits.masked_fill(~positive, -1e4), -1)
    return (all_logits - positive_logits)[eligible].mean(), eligible
