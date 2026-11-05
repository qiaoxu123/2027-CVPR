"""Geometry-only utilities for evaluating pretrained AerialVG on CityNav crops.

No AerialVG code/weights are vendored. The mapping follows the four-corner
orthographic CityNav renderer; it is not a perspective-depth reconstruction.
Ground truth is used ONLY by the caller after inference, never for proposals.
"""
from __future__ import annotations

import cv2
import numpy as np


def image_to_world_matrix(footprint_world, height: int, width: int):
    world = np.asarray(footprint_world, dtype=np.float32)
    if world.shape != (4, 2) or not np.isfinite(world).all():
        raise ValueError("Expected four finite CityNav footprint corners")
    if height < 2 or width < 2:
        raise ValueError("Image must have at least two pixels per dimension")
    source = np.float32([
        [0, 0], [width - 1, 0],
        [width - 1, height - 1], [0, height - 1],
    ])
    return cv2.getPerspectiveTransform(source, world)


def image_box_to_world(box_cxcywh, footprint_world, height: int, width: int):
    """Project a normalized predicted box center into CityNav world XY."""
    box = np.asarray(box_cxcywh, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError("A finite normalized cxcywh box is required")
    if np.any(box[:2] < 0) or np.any(box[:2] > 1) or np.any(box[2:] < 0):
        raise ValueError("Invalid normalized box")
    point = np.float32([[[box[0] * (width - 1),
                          box[1] * (height - 1)]]])
    xy = cv2.perspectiveTransform(
        point, image_to_world_matrix(footprint_world, height, width)
    )[0, 0]
    return [float(xy[0]), float(xy[1])]


def target_is_in_current_view(target_world_xy, footprint_world):
    """Post-inference scoring metadata only; never use for proposal filtering."""
    target = np.asarray(target_world_xy, dtype=np.float32)
    polygon = np.asarray(footprint_world, dtype=np.float32)
    if target.shape != (2,) or polygon.shape != (4, 2):
        raise ValueError("Invalid target or footprint geometry")
    return cv2.pointPolygonTest(polygon, tuple(map(float, target)), False) >= 0


def rank_aerialvg_boxes(boxes, logits, footprint_world, height, width,
                        *, threshold=0.35, top_k=5):
    """Rank ORIGINAL AerialVG decoder outputs with its standard sigmoid score.

    This is an offline detector baseline, not a drop-in compatible relation head.
    """
    import torch

    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("Expected Nx4 decoder boxes")
    if logits.ndim != 2 or logits.shape[0] != boxes.shape[0]:
        raise ValueError("Prediction/logit row count mismatch")
    if not 0 <= threshold <= 1 or top_k < 1:
        raise ValueError("Invalid threshold or top-k")
    scores = logits.float().sigmoid().amax(dim=-1)
    if not torch.isfinite(scores).all() or not torch.isfinite(boxes).all():
        raise ValueError("Nonfinite AerialVG predictions")
    selected = torch.where(scores >= threshold)[0]
    selected = selected[torch.argsort(scores[selected], descending=True)[:top_k]]
    result = []
    for i in selected.tolist():
        box = boxes[i].detach().float().cpu().tolist()
        try:
            xy = image_box_to_world(box, footprint_world, height, width)
        except ValueError:
            continue
        result.append(dict(query=int(i), score=float(scores[i]), box_cxcywh=box,
                           world_xy=xy))
    return result


def score_frame(predictions, target_world_xy, *, target_visible):
    """All eligible frames, including empty predictions, remain in denominator."""
    target = np.asarray(target_world_xy, dtype=np.float64)
    errors = [float(np.linalg.norm(
        np.asarray(p["world_xy"], dtype=np.float64) - target
    )) for p in predictions]
    return dict(
        count=len(errors), target_visible=bool(target_visible),
        top1_NE_m=errors[0] if errors else None,
        top1_R20=bool(errors and errors[0] <= 20.0),
        oracle_topk_R20=bool(any(e <= 20.0 for e in errors)),
    )
