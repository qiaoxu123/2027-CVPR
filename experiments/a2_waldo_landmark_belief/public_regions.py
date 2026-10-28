"""Exact JSON-safe public map surfaces for an opt-in A2 revision.

This module receives public coordinate rings only. It neither resolves names
nor accepts destinations/labels, and is not wired into the running V3 policy.
"""
from collections.abc import Mapping, Sequence

import numpy as np
from shapely.geometry import Point, Polygon, mapping, shape
from shapely.errors import ShapelyError
from shapely.ops import unary_union


VERSION = 'public-region-v1'


def _valid_area(geometry):
    if geometry.geom_type not in ('Polygon', 'MultiPolygon'):
        raise ValueError('Public region must contain polygon area')
    if geometry.is_empty or not geometry.is_valid or not np.isfinite(geometry.area) or geometry.area <= 0:
        raise ValueError('Invalid or empty public region')
    # Check every ring, not only bounds: NaNs can survive malformed inputs.
    parts = [geometry] if geometry.geom_type == 'Polygon' else list(geometry.geoms)
    for polygon in parts:
        for ring in [polygon.exterior, *polygon.interiors]:
            coordinates = np.asarray(ring.coords)
            if coordinates.ndim != 2 or coordinates.shape[1] != 2 or not np.isfinite(coordinates).all():
                raise ValueError('Public region requires finite XY coordinates')
    return geometry


def region_from_contours(contours: Sequence) -> dict:
    """Exact union of valid public exterior contours, without hull/repair."""
    polygons = []
    for contour in contours:
        try:
            points = np.asarray(contour, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError('Malformed public contour') from error
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
            raise ValueError('Public contour requires at least three finite XY points')
        polygons.append(_valid_area(Polygon(points)))
    if not polygons:
        raise ValueError('Public region requires at least one contour')
    geometry = _valid_area(unary_union(polygons))
    return dict(version=VERSION, geometry=mapping(geometry))


def region_shape(region: Mapping):
    """Validate a serialized region and recover all holes and components."""
    if not isinstance(region, Mapping) or region.get('version') != VERSION:
        raise ValueError('Unknown public region version')
    serialized = region.get('geometry')
    if not isinstance(serialized, Mapping) or serialized.get('type') not in ('Polygon', 'MultiPolygon'):
        raise ValueError('Public region requires Polygon or MultiPolygon geometry')
    try:
        geometry = shape(serialized)
    except (KeyError, TypeError, ValueError, IndexError, ShapelyError) as error:
        raise ValueError('Malformed public region geometry') from error
    return _valid_area(geometry)


def region_statistics(region: Mapping, point: Sequence) -> dict:
    """Union centroid/area and signed boundary distance; holes are outside."""
    geometry = region_shape(region)
    coordinates = np.asarray(point, dtype=np.float64)
    if coordinates.shape != (2,) or not np.isfinite(coordinates).all():
        raise ValueError('Query point must be finite XY')
    query = Point(coordinates)
    inside = bool(geometry.covers(query))
    distance = float(geometry.boundary.distance(query))
    return dict(area=float(geometry.area), center=[float(geometry.centroid.x), float(geometry.centroid.y)],
                signed_distance=distance if inside else -distance, inside=inside)
