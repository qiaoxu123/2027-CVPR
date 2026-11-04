"""Named-public-region weak supervision, never pixel GT or visibility proof.

Labels depend on projected region coverage and uncertainty margins only. Do not
derive a Null/absence target when no positive observed window can be confirmed.
"""
import numpy as np
import shapely
from shapely.geometry import Polygon

from .landmark_projection import frame_to_world


def patch_world_windows(footprint_world):
    """Reproduce the cached 1/3/7 patch supports, including clipped borders."""
    gy, gx = np.meshgrid(np.arange(1, 24, 2), np.arange(1, 24, 2), indexing='ij')
    centers = np.stack((gx.ravel(), gy.ravel()), -1)
    pixels = []
    for x, y in centers:
        scales = []
        for radius in (0, 1, 3):
            x0, y0 = max(0, x-radius)*16, max(0, y-radius)*16
            x1, y1 = min(24, x+radius+1)*16-1, min(24, y+radius+1)*16-1
            scales.append([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        pixels.append(scales)
    return frame_to_world(np.array(pixels).reshape(-1, 2), footprint_world, (384, 384)).reshape(144, 3, 4, 2)


def reference_region_labels(world_windows, public_contour, valid, *, margin_m=2.,
                            positive_coverage=.9, negative_coverage=.01):
    windows = np.asarray(world_windows, float)
    valid = np.asarray(valid)
    if windows.ndim != 3 or windows.shape[1:] != (4, 2) or valid.shape != (len(windows),) or valid.dtype != bool:
        raise ValueError('aligned world windows and explicit validity required')
    if not np.isfinite(windows).all() or not np.isfinite(margin_m) or margin_m < 0 or not 0 <= negative_coverage < positive_coverage <= 1:
        raise ValueError('finite geometry and valid uncertainty margins required')
    labels = np.full(len(windows), -1, np.int8)
    coverage = np.zeros((len(windows), 2), np.float32)
    contour = np.asarray(public_contour, float)
    if contour.ndim != 2 or contour.shape[1] != 2 or len(contour) < 3 or not np.isfinite(contour).all():
        return labels, coverage
    region = Polygon(contour)
    if not region.is_valid or region.is_empty or region.area <= 0:
        return labels, coverage
    areas = shapely.polygons(windows)
    size = shapely.area(areas)
    supported = valid & shapely.is_valid(areas) & (size > 1e-6)
    inner, outer = region.buffer(-margin_m), region.buffer(margin_m)
    for index, polygon in enumerate((inner, outer)):
        if not polygon.is_empty:
            coverage[supported, index] = np.clip(
                shapely.area(shapely.intersection(areas[supported], polygon))/size[supported], 0, 1)
    labels[supported & (coverage[:, 1] <= negative_coverage)] = 0
    labels[supported & (coverage[:, 0] >= positive_coverage)] = 1
    return labels, coverage
