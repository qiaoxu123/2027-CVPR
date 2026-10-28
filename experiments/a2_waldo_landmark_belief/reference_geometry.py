"""Shared new-version public geometry; legacy single-contour path is exact."""
import math
from collections.abc import Mapping, Sequence

import cv2
import numpy as np

from .core import _center, _polygon, box_landmark_geometry
from .public_regions import region_shape, region_statistics


def public_reference_center(reference: Mapping) -> np.ndarray:
    """One centroid definition for public features and private TRAIN labels."""
    if 'region' in reference:
        center = region_shape(reference['region']).centroid
        return np.asarray([center.x, center.y], dtype=np.float64)
    contour = _polygon(reference['contour'])
    if contour is None:
        raise ValueError('Invalid singleton public reference contour')
    return _center(contour)


def box_reference_geometry(box_polygons: Sequence, references: Sequence[Mapping],
                           box_confidences: Sequence[float], *, radius_m: float = 40.,
                           box_height_m: Sequence[float | None] | None = None,
                           landmark_height_m: Sequence[float | None] | None = None):
    """Same twelve feature slots as V3; exact union geometry for grouped roads.

    Group heights are intentionally missing even when live depth is available,
    matching what can faithfully be rebuilt from the old training sources.
    """
    legacy_contours = [r['contour'] if 'region' not in r else [] for r in references]
    out, valid = box_landmark_geometry(box_polygons, legacy_contours, box_confidences,
        radius_m=radius_m, box_height_m=box_height_m, landmark_height_m=landmark_height_m)
    for j, reference in enumerate(references):
        if 'region' not in reference:
            continue
        if reference.get('height_policy') != 'missing_for_public_road_group':
            raise ValueError('Grouped reference height policy must be explicitly missing')
        # Validate even with no candidates, so bad reference data cannot hide
        # behind an empty detector output.
        region_shape(reference['region'])
        for i, polygon in enumerate(box_polygons):
            box = _polygon(polygon)
            confidence = float(box_confidences[i])
            if box is None or not np.isfinite(confidence) or not 0 <= confidence <= 1:
                continue
            center = _center(box)
            info = region_statistics(reference['region'], center)
            delta = center.astype(np.float64)-np.asarray(info['center'])
            distance = float(np.linalg.norm(delta))
            area = abs(cv2.contourArea(box))
            out[i,j] = [delta[0]/radius_m, delta[1]/radius_m, distance/radius_m,
                delta[1]/max(distance,1e-6), delta[0]/max(distance,1e-6),
                math.log1p(distance)/math.log1p(radius_m), info['signed_distance']/radius_m,
                float(info['inside']), math.log(max(area,1e-6)/info['area'])/5,
                confidence, 0, 0]
            valid[i,j] = True
    return out, valid
