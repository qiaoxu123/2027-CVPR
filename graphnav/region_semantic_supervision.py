"""TRAIN annotation-side land-use proxies, NOT instance segmentation truth.

Build/parking/ground describe regions, not the car identities inside them.
Outside all registered regions and cross-class overlaps always stay unknown.
"""
import numpy as np
import shapely
from shapely.geometry import Polygon

REGION_CLASSES = ('Building', 'Parking', 'Ground')


def semantic_region_labels(world_windows, contours, valid, *, margin_m=2.):
    windows = np.asarray(world_windows, float); valid = np.asarray(valid)
    if windows.ndim != 3 or windows.shape[1:] != (4, 2) or valid.shape != (len(windows),) or valid.dtype != bool:
        raise ValueError('aligned public world windows and validity required')
    if not np.isfinite(windows).all() or not np.isfinite(margin_m) or margin_m < 0:
        raise ValueError('finite geometry and nonnegative uncertainty margin required')
    patch = shapely.polygons(windows); size = shapely.area(patch)
    supported = valid & shapely.is_valid(patch) & (size > 1e-6)
    coverage = np.zeros((len(windows), len(REGION_CLASSES), 2), np.float32)
    for index, name in enumerate(REGION_CLASSES):
        regions = []
        for contour in contours.get(name, []):
            xy = np.asarray(contour, float)
            if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3 or not np.isfinite(xy).all():
                continue
            polygon = Polygon(xy)
            if polygon.is_valid and polygon.area > 0:
                regions.append(polygon)
        if not regions:
            continue
        union = shapely.union_all(regions)
        for side, region in enumerate((union.buffer(-margin_m), union.buffer(margin_m))):
            if not region.is_empty:
                coverage[supported, index, side] = np.clip(
                    shapely.area(shapely.intersection(patch[supported], region))/size[supported], 0, 1)
    positive = coverage[:, :, 0] >= .9
    labels = np.full(len(windows), -1, np.int8)
    for index in range(len(REGION_CLASSES)):
        other = np.arange(len(REGION_CLASSES)) != index
        accepted = supported & positive[:, index] & (coverage[:, other, 1] <= .01).all(-1)
        labels[accepted] = index
    return labels, coverage
