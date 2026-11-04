"""Public-map-only geometry around a named landmark (diagnostics, NOT policy).

The nearby-object table is read only after the named-goal arrival diagnostic.
It does not use CityNav target GT, teacher trajectories or future observations.
World XY are the public map's east/north axes; do not infer camera-left/right.
"""
import math

import numpy as np
from shapely.geometry import Polygon


def nearby_public_geometry(objects, landmark_polygon, excluded_ids, *,
                           radius_m=100., limit=12):
    if radius_m <= 0 or limit < 1:
        raise ValueError("Invalid neighborhood query")
    excluded = {str(i) for i in excluded_ids}
    anchor = landmark_polygon.centroid
    result = []
    for object_id, obj in objects.items():
        if str(object_id) in excluded:
            continue
        contour = np.asarray(obj.get("contour", []), dtype=float)
        if (contour.ndim != 2 or contour.shape[1:] != (2,)
                or len(contour) < 3 or not np.isfinite(contour).all()):
            continue
        polygon = Polygon(contour)
        if not polygon.is_valid or polygon.is_empty or polygon.area <= 0:
            continue
        distance = float(landmark_polygon.distance(polygon))
        if distance > radius_m:
            continue
        center = polygon.centroid
        dx = float(center.x - anchor.x)
        dy = float(center.y - anchor.y)
        result.append({
            "public_id": str(object_id),
            "name": str(obj.get("name") or ""),
            "type": str(obj.get("object_type") or "Unknown"),
            "centroid_xy": [float(center.x), float(center.y)],
            "relative_centroid_dx_m": dx,
            "relative_centroid_dy_m": dy,
            "centroid_bearing_deg_from_east": float(math.degrees(math.atan2(dy, dx))),
            "polygon_boundary_distance_m": distance,
            "intersects_public_polygon": bool(landmark_polygon.intersects(polygon)),
            "contains_or_within": bool(
                landmark_polygon.contains(polygon) or polygon.contains(landmark_polygon)
            ),
        })
    result.sort(key=lambda row: (
        row["polygon_boundary_distance_m"],
        row["type"], row["public_id"],
    ))
    return result[:limit]
