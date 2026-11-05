"""Offline public-map relations. None of these functions are policy inputs."""
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union


def legal_polygon(contour):
    a = np.asarray(contour, float)
    if a.ndim != 2 or a.shape[1:] != (2,) or len(a) < 3 or not np.isfinite(a).all():
        return None
    p = Polygon(a)
    return p if p.is_valid and not p.is_empty and p.area > 0 else None


def neighbors(objects, anchor, excluded, radius=100.):
    """All objects, no inherited 12-neighbor display cap; metric boundary distance."""
    result = []
    for key, obj in objects.items():
        polygon = legal_polygon(obj.get('contour', []))
        if str(key) in excluded or polygon is None or anchor.distance(polygon) > radius:
            continue
        result.append((str(key), obj, polygon))
    return sorted(result, key=lambda x: (anchor.distance(x[2]), x[0]))


def relation(first, second):
    """Direction is centroid World XY, NOT assumed geographic or camera-relative."""
    delta = np.array(second.centroid.coords[0]) - first.centroid.coords[0]
    return dict(boundary_distance_m=float(first.distance(second)),
                delta_centroid_world_xy=delta.tolist(),
                bearing_deg_from_world_plus_x=float(np.degrees(np.arctan2(delta[1], delta[0]))),
                intersects=bool(first.intersects(second)), touches=bool(first.touches(second)),
                first_covers_second=bool(first.covers(second)),
                second_covers_first=bool(second.covers(first)),
                overlap_fraction_of_second=float(first.intersection(second).area / second.area),
                evidence='public_polygon_geometry_only')


def road_enclosures(roads):
    """Only exact holes of a road-surface union; no buffer/gap-filling fiction.

    Absence of holes is UNKNOWN: missing road fragments may prevent closure.
    A hole is a map-topology proxy, never evidence of RGB recognition.
    """
    if not roads:
        return []
    union = unary_union([p for _, p in roads])
    parts = [union] if union.geom_type == 'Polygon' else getattr(union, 'geoms', [])
    holes = []
    for part in parts:
        if part.geom_type != 'Polygon':
            continue
        for ring in part.interiors:
            hole = Polygon(ring)
            adjacent = [key for key, road in roads if road.distance(hole) < 1e-6]
            holes.append(dict(contour=list(map(list, hole.exterior.coords)), area_m2=hole.area,
                              adjacent_road_ids=adjacent, evidence='exact_public_road_union_hole'))
    return holes


def enclosure_road_names(road_ids, objects):
    """Several fragments of the same named road are not several roads."""
    from graphnav.landmark_alias import canonical_name
    names = set()
    unnamed = []
    for key in road_ids:
        name = canonical_name(objects.get(key, {}).get('name', ''))
        if name:
            names.add(name)
        else:
            unnamed.append(key)
    return dict(distinct_named_roads=sorted(names), unnamed_fragment_ids=unnamed,
                distinct_named_road_count=len(names),
                identity_qualification='map_name_groups_only_not_RGB_identity')
