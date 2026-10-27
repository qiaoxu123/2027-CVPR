"""Source-faithful local ROI scales extracted ONLY from the current RGB view.

The 384px result is digital resampling of already-observed source pixels:
it CANNOT reveal small objects absent from the original image. Real
improvement in physical sampling requires a new observation at another pose.
"""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np

from .landmark_projection import world_to_frame


@dataclass(frozen=True)
class ObservedScale:
    bgr: np.ndarray
    support: np.ndarray
    native_span_px: float
    valid_fraction: float
    extent_m: float
    center_world: tuple[float, float]


def extract_observed_scale(observation, center_world, *, extent_m,
                           output_size=384, min_native_span_px=24,
                           min_valid_fraction=.9):
    """Crop square world-metric ROI only when wholly inside current FOV.

    Uses current image+depth support, NOT underlying orthophoto, target ID or
    future frame. Returns None if crop is too tiny, clipped, or invalid.
    """
    if extent_m <= 0 or output_size < 32 or min_native_span_px < 2:
        raise ValueError("invalid ROI scale parameters")
    if not 0 < min_valid_fraction <= 1:
        raise ValueError("invalid support threshold")
    center = np.asarray(center_world, dtype=np.float32)
    bgr = np.asarray(observation["bgr"])
    valid_depth = np.asarray(observation["depth_valid"], dtype=bool)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("invalid public target candidate center")
    if bgr.ndim != 3 or bgr.shape[2] != 3 or valid_depth.shape != bgr.shape[:2]:
        raise ValueError("observed RGB/Depth dimension mismatch")
    h, w = bgr.shape[:2]
    radius = extent_m / 2.
    world_quad = center + np.array(
        [[-radius, -radius], [radius, -radius],
         [radius, radius], [-radius, radius]], dtype=np.float32,
    )
    corners = world_to_frame(world_quad, observation["footprint_world"], bgr.shape)
    if not np.isfinite(corners).all():
        return None
    if np.any(corners[:, 0] < 0) or np.any(corners[:, 0] > w - 1):
        return None
    if np.any(corners[:, 1] < 0) or np.any(corners[:, 1] > h - 1):
        return None
    native_area = abs(float(cv2.contourArea(corners)))
    native_span = native_area ** .5
    if native_span < min_native_span_px:
        return None
    dst = np.array(
        [[0, 0], [output_size - 1, 0],
         [output_size - 1, output_size - 1], [0, output_size - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    source_valid = (bgr.max(-1) > 0) & valid_depth
    output_valid = cv2.warpPerspective(
        source_valid.astype(np.uint8), transform,
        (output_size, output_size), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    ).astype(bool)
    fraction = float(output_valid.mean())
    if fraction < min_valid_fraction:
        return None
    output = cv2.warpPerspective(
        bgr, transform, (output_size, output_size),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    output[~output_valid] = 0
    return ObservedScale(
        bgr=output, support=output_valid, native_span_px=native_span,
        valid_fraction=fraction, extent_m=float(extent_m),
        center_world=(float(center[0]), float(center[1])),
    )


def multiscale_observed_crops(observation, center_world, *,
                              extents_m=(32., 64., 128.),
                              output_size=384):
    """Optional offline retrieval ablation; never auto-encode extra images."""
    result = {}
    for extent in extents_m:
        if extent <= 0:
            raise ValueError("all world-metric ROI extents must be positive")
        item = extract_observed_scale(
            observation, center_world, extent_m=float(extent),
            output_size=output_size,
        )
        if item is not None:
            result[float(extent)] = item
    return result
