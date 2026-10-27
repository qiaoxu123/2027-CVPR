"""Geometry-only registration between CityRefer world polygons and *observed* RGB.

This yields projective ROI annotations, NOT ground-truth pixel segmentation.
Works for CityNav's simulated nadir camera with known TL/TR/BR/BL footprint.
For a real perspective camera use calibrated 3D projection/visibility instead.
No target positions, future poses, orthophoto, or trajectory labels are read.
"""

from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np
from .semantic_geometry import valid_surface_polygon


@dataclass(frozen=True)
class LandmarkROI:
    object_id: str
    name: str
    object_type: str
    contour_px: np.ndarray
    mask: np.ndarray
    visible_fraction: float
    projected_area_px: float
    visible_pixels: int
    bbox_xyxy: tuple[int, int, int, int]


def world_to_frame(world_xy, footprint_world, image_shape):
    """Project world XY using observed frame corner homography, not map bounds."""
    quad = np.asarray(footprint_world, np.float32)
    xy = np.asarray(world_xy, np.float32)
    h, w = image_shape[:2]
    if quad.shape != (4, 2) or xy.ndim != 2 or xy.shape[-1] != 2:
        raise ValueError("invalid world polygon or footprint")
    if h < 2 or w < 2 or not np.isfinite(quad).all() or not np.isfinite(xy).all():
        raise ValueError("invalid image coordinates")
    if abs(cv2.contourArea(quad)) < 1e-5:
        raise ValueError("degenerate camera footprint")
    pixels = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], np.float32)
    H = cv2.getPerspectiveTransform(quad, pixels)
    return cv2.perspectiveTransform(xy[None], H)[0]


def frame_to_world(pixels, footprint_world, image_shape):
    quad = np.asarray(footprint_world, np.float32)
    xy = np.asarray(pixels, np.float32)
    h, w = image_shape[:2]
    screen = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], np.float32)
    if quad.shape != (4, 2) or xy.ndim != 2 or xy.shape[-1] != 2:
        raise ValueError("invalid projection")
    H = cv2.getPerspectiveTransform(screen, quad)
    return cv2.perspectiveTransform(xy[None], H)[0]


def project_landmark(
    landmark, footprint_world, image_shape, *,
    min_visible=0.50, min_pixels=32,
):
    """Create an observed-only ROI mask; reject tiny/clipped instances.

    Expects a landmark contour in WORLD meters. Coverage is geometric
    projected polygon area, not visual agreement with a semantic mask.
    """
    xy = np.asarray(landmark["contour"], np.float32)
    if not valid_surface_polygon(xy):
        return None
    try:
        px = world_to_frame(xy, footprint_world, image_shape)
    except (ValueError, cv2.error):
        return None
    if not np.isfinite(px).all() or np.max(np.abs(px)) > 1e6:
        return None
    area = abs(float(cv2.contourArea(px)))
    if area < min_pixels:
        return None
    h, w = image_shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.rint(px).astype(np.int32)], 1)
    visible = int(mask.sum())
    fraction = min(1.0, visible / max(area, 1e-6))
    if fraction < min_visible or visible < min_pixels:
        return None
    ys, xs = np.where(mask != 0)
    if len(xs) == 0:
        return None
    return LandmarkROI(
        object_id=str(landmark["id"]),
        name=str(landmark.get("name", "")),
        object_type=str(landmark.get("object_type", "Unknown")),
        contour_px=px,
        mask=mask,
        visible_fraction=fraction,
        projected_area_px=area,
        visible_pixels=visible,
        bbox_xyxy=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
    )


def make_roi_image(bgr, roi: LandmarkROI, output_size=224, context=0.25):
    """Keep RGB context outside the mask; use it only for supervision/metadata.

    Hard-zeroing untrusted polygon boundaries would corrupt visual features.
    """
    if not 0 <= context <= 2 or output_size < 32:
        raise ValueError("invalid crop configuration")
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = roi.bbox_xyxy
    pad = int(max(x1-x0, y1-y0) * context)
    x0, x1 = max(0, x0-pad), min(w, x1+pad)
    y0, y1 = max(0, y0-pad), min(h, y1+pad)
    cropped = bgr[y0:y1, x0:x1]
    if cropped.size == 0:
        raise ValueError("empty landmark crop")
    # Letterbox to preserve shape and surrounding spatial context.
    height, width = cropped.shape[:2]
    factor = output_size / max(height, width)
    new_h, new_w = max(1, round(height*factor)), max(1, round(width*factor))
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((output_size, output_size, 3), np.uint8)
    top, left = (output_size-new_h)//2, (output_size-new_w)//2
    canvas[top:top+new_h, left:left+new_w] = resized
    return canvas


def draw_projected_roi(bgr, roi: LandmarkROI):
    result = bgr.copy()
    pts = np.rint(roi.contour_px).astype(np.int32)
    cv2.polylines(result, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    x0, y0, _, _ = roi.bbox_xyxy
    cv2.putText(result, f"{roi.object_type}:{roi.name}"[:48],
                (max(0, x0), max(10, y0-4)), cv2.FONT_HERSHEY_SIMPLEX,
                .4, (0, 255, 0), 1, cv2.LINE_AA)
    return result
