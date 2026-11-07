"""Observed-only candidate ROI utilities for the frozen V2 diagnostic pilot.

All crops are warped from a single 384px RGB-D frame rendered at a logged
policy pose. No orthomosaic, target identity, contour, or future pose is an
input to this module.
"""
from __future__ import annotations

import cv2
import numpy as np

from .landmark_projection import world_to_frame


def observed_candidate_roi(observation, center_xy, extent_m, *, output_size=384):
    """Return a masked world-axis ROI from one current observed frame.

    A crop is valid only when its center is depth-valid and at least 85% of
    sampled pixels have valid support. The support mask is applied before it
    leaves the renderer, so pixels outside the current footprint are unknown.
    """
    if extent_m <= 0 or output_size < 32:
        raise ValueError("positive extent and output size >=32 required")
    rgb = np.asarray(observation["bgr"])
    depth_valid = np.asarray(observation["depth_valid"], dtype=bool)
    footprint = np.asarray(observation["footprint_world"], dtype=np.float32)
    center = np.asarray(center_xy, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError("observation RGB must be uint8 BGR")
    if depth_valid.shape != rgb.shape[:2] or footprint.shape != (4, 2):
        raise ValueError("RGB-D support or footprint shape mismatch")
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("finite candidate center required")

    h, w = rgb.shape[:2]
    half = float(extent_m) / 2.0
    quad_world = np.asarray([
        [center[0] - half, center[1] + half],
        [center[0] + half, center[1] + half],
        [center[0] + half, center[1] - half],
        [center[0] - half, center[1] - half],
    ], dtype=np.float32)
    try:
        quad_px = world_to_frame(quad_world, footprint, rgb.shape)
        center_px = world_to_frame(center[None], footprint, rgb.shape)[0]
    except (ValueError, cv2.error):
        return {"valid": False, "support": 0.0, "image": None,
                "center_px": None, "quad_px": None}
    if (not np.isfinite(quad_px).all() or not np.isfinite(center_px).all()
            or np.max(np.abs(quad_px)) > 1e6):
        return {"valid": False, "support": 0.0, "image": None,
                "center_px": None, "quad_px": None}
    cx, cy = np.rint(center_px).astype(int)
    if not (0 <= cx < w and 0 <= cy < h):
        return {"valid": False, "support": 0.0, "image": None,
                "center_px": center_px.tolist(), "quad_px": quad_px.tolist()}
    y0, y1 = max(0, cy - 1), min(h, cy + 2)
    x0, x1 = max(0, cx - 1), min(w, cx + 2)
    center_support = float(depth_valid[y0:y1, x0:x1].mean())
    dst = np.float32([[0, 0], [output_size - 1, 0],
                      [output_size - 1, output_size - 1], [0, output_size - 1]])
    matrix = cv2.getPerspectiveTransform(quad_px.astype(np.float32), dst)
    crop = cv2.warpPerspective(rgb, matrix, (output_size, output_size),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    support_mask = cv2.warpPerspective(depth_valid.astype(np.uint8), matrix,
                                       (output_size, output_size),
                                       flags=cv2.INTER_NEAREST,
                                       borderMode=cv2.BORDER_CONSTANT)
    mask = support_mask.astype(bool)
    support = float(mask.mean())
    crop[~mask] = 0
    return {"valid": bool(center_support >= 0.5 and support >= 0.85),
            "support": support, "center_support": center_support,
            "image": crop, "center_px": center_px.tolist(),
            "quad_px": quad_px.tolist()}


def first_causal_comparable_frame(poses, observe_at_pose, candidates,
                                  *, extents=(20.0, 40.0), min_candidates=2):
    """Find the first logged pose where Top-1 and an alternative share support.

    `observe_at_pose` must render only the supplied pose. Candidate centers are
    model outputs. GT is deliberately not accepted by the function signature.
    """
    if len(candidates) < min_candidates:
        return None
    for step, pose in enumerate(poses):
        observation = observe_at_pose(pose)
        crops = {}
        valid_by_candidate = []
        for rank, candidate in enumerate(candidates):
            ext = {}
            for extent in extents:
                ext[extent] = observed_candidate_roi(
                    observation, candidate["xy"], extent)
            crops[rank] = ext
            valid_by_candidate.append(all(ext[e]["valid"] for e in extents))
        jointly_valid = [i for i, value in enumerate(valid_by_candidate) if value]
        if 0 in jointly_valid and len(jointly_valid) >= min_candidates:
            return {"step": step, "pose": np.asarray(pose).tolist(),
                    "observation": observation, "crops": crops,
                    "jointly_valid": jointly_valid}
    return None


def conservative_visual_rerank(candidate_scores, valid_indices, *, baseline_index=0):
    """Only switch if one alternative beats Top-1 in every frozen score view.

    `candidate_scores` has shape [K, views]. This is a predeclared consensus
    abstention rule, not a learned or calibrated confidence estimate.
    """
    scores = np.asarray(candidate_scores, dtype=np.float64)
    valid = [int(i) for i in valid_indices]
    if scores.ndim != 2 or not (0 <= baseline_index < len(scores)):
        raise ValueError("candidate scores must be [K,views]")
    if len(valid) < 2 or baseline_index not in valid or not np.isfinite(scores[valid]).all():
        return int(baseline_index), False
    baseline = scores[baseline_index]
    challengers = [i for i in valid if i != baseline_index
                   and bool(np.all(scores[i] > baseline))]
    if not challengers:
        return int(baseline_index), False
    winner = max(challengers, key=lambda i: float(scores[i].mean()))
    return int(winner), True
