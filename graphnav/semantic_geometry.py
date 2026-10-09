"""Explicit, identity-safe geometric relations between whole landmarks and roads.

A building polygon is ONE entity, never a node per contour vertex.
Roads are LINE entities. Road-road topology is inferred only when mapped
segments actually touch/cross within a physical tolerance; we do not create
road topology by generic k-NN.
XY: (east, north). Lengths computed in *meters*, never normalized units.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import numpy as np


@dataclass(frozen=True)
class SpatialRelation:
    kind: str
    anchors: tuple[tuple[int, ...], ...]
    # Each anchor group may contain multiple IDs with identical names.
    # Evaluation uses an OR/max across ambiguous matches, never GT-nearest.


DIRECTIONS = {
    "east": (1.0, 0.0), "west": (-1.0, 0.0),
    "north": (0.0, 1.0), "south": (0.0, -1.0),
    "northeast": (1.0, 1.0), "northwest": (-1.0, 1.0),
    "southeast": (1.0, -1.0), "southwest": (-1.0, -1.0),
}
RELATIONS = tuple(DIRECTIONS) + ("near", "between", "along")
NODE_TYPES = {"building": 0, "road": 1, "relation": 2}
EDGE_TYPES = {
    "self": 0, "buildings_near": 1, "buildings_overlap": 2,
    "building_road": 3, "roads_connected": 4,
    "relation_anchor": 5, "anchor_relation": 6,
}


def canonical(text: str) -> str:
    text = str(text).casefold().replace("’", "'").replace("-", " ")
    return " ".join(re.findall(r"[a-z0-9]+", text))


def polygon_center(vertices):
    pts = np.asarray(vertices, np.float32)
    if len(pts) == 0:
        raise ValueError("empty geometry")
    return pts.mean(axis=0)


def edges_of(points, closed=False):
    x = np.asarray(points, dtype=np.float32)
    if len(x) < 2:
        return []
    result = [(x[i], x[i + 1]) for i in range(len(x) - 1)]
    if closed and len(x) > 2:
        result.append((x[-1], x[0]))
    return result


def distance_to_segment(point, a, b):
    p, a, b = (np.asarray(x, np.float32) for x in (point, a, b))
    v = b - a
    t = np.clip(np.sum((p - a) * v, axis=-1) / max(float(v @ v), 1e-12), 0., 1.)
    return np.linalg.norm(p - (a + t[..., None] * v), axis=-1)


def point_inside_polygon(point, polygon):
    p = np.asarray(point, np.float32)
    polygon = np.asarray(polygon, np.float32)
    hits = np.zeros(p.shape[:-1], dtype=bool)
    x, y = p[..., 0], p[..., 1]
    for a, b in edges_of(polygon, closed=True):
        ay, by = float(a[1]), float(b[1])
        if abs(by - ay) < 1e-12:
            continue
        intersects = ((ay > y) != (by > y)) & (
            x < (float(b[0]) - float(a[0])) * (y - ay) / (by - ay) + a[0]
        )
        hits ^= intersects
    return hits


def geometry_distance(point, geometry, is_road=False):
    """Distance to road polyline OR polygon exterior, zero when inside polygon."""
    p = np.asarray(point, np.float32)
    segments = edges_of(geometry, closed=not is_road)
    if not segments:
        return np.full(p.shape[:-1], np.inf, np.float32)
    d = np.minimum.reduce([distance_to_segment(p, a, b) for a, b in segments])
    if not is_road:
        d = np.where(point_inside_polygon(p, geometry), 0., d)
    return d


def _segments_intersect(a, b, c, d):
    def ccw(p, q, r):
        return float(np.cross(q - p, r - p))  # scalar 2-D
    c1, c2, c3, c4 = ccw(a, b, c), ccw(a, b, d), ccw(c, d, a), ccw(c, d, b)
    if (c1 * c2 < 0) and (c3 * c4 < 0):
        return True
    for p, s, t in ((a, c, d), (b, c, d), (c, a, b), (d, a, b)):
        if abs(float(np.cross(t - s, p - s))) < 1e-5 and (
            min(s[0], t[0]) - 1e-5 <= p[0] <= max(s[0], t[0]) + 1e-5
        ) and (min(s[1], t[1]) - 1e-5 <= p[1] <= max(s[1], t[1]) + 1e-5):
            return True
    return False


def geometry_pair_distance(a, b, a_road=False, b_road=False):
    """Segment-to-segment distance in meters, including overlaps/intersections."""
    ea, eb = edges_of(a, closed=not a_road), edges_of(b, closed=not b_road)
    if not ea or not eb:
        return float("inf")
    for x, y in ea:
        for u, v in eb:
            if _segments_intersect(x, y, u, v):
                return 0.
    candidates = []
    for x, y in ea:
        for u, v in eb:
            candidates.extend((
                float(distance_to_segment(x, u, v)),
                float(distance_to_segment(y, u, v)),
                float(distance_to_segment(u, x, y)),
                float(distance_to_segment(v, x, y)),
            ))
    # Building containment has 0 distance even if no edges intersect.
    if not a_road and point_inside_polygon(np.asarray(b)[0], a).item():
        return 0.
    if not b_road and point_inside_polygon(np.asarray(a)[0], b).item():
        return 0.
    return min(candidates)


def _reference_groups(entities):
    ids = {}
    for i, entity in enumerate(entities):
        name = canonical(entity.get("name", ""))
        if name:
            ids.setdefault(name, []).append(i)
    return ids


def parse_spatial_relations(instruction: str, entities: list[dict]) -> list[SpatialRelation]:
    """Conservative explicit relation extraction, anchored to actual map names.

    No ungrounded relation node: 'east' without a matched reference is ignored.
    'between' requires TWO different grounded names with nearby conjunction.
    Multi-matched names stay ambiguous sets, never resolved using target XY.
    """
    text = canonical(instruction)
    group = _reference_groups(entities)
    if not text or not group:
        return []
    names = sorted(group, key=len, reverse=True)
    result, used = [], set()
    for kind in DIRECTIONS:
        for name in names:
            pattern = rf"\b{kind}\s+of\s+(?:the\s+)?{re.escape(name)}\b"
            if re.search(pattern, text):
                key = (kind, name)
                if key not in used:
                    result.append(SpatialRelation(kind, (tuple(group[name]),)))
                    used.add(key)
    for kind, trigger in (
        ("near", r"(?:near|beside|close to|next to)"),
        ("along", r"(?:along|alongside)"),
    ):
        for name in names:
            pattern = rf"\b{trigger}\s+(?:the\s+)?{re.escape(name)}\b"
            if re.search(pattern, text):
                key = (kind, name)
                if key not in used:
                    result.append(SpatialRelation(kind, (tuple(group[name]),)))
                    used.add(key)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            # 'between A and B' as well as its reverse ordering.
            for left, right in ((first, second), (second, first)):
                pattern = rf"\bbetween\s+(?:the\s+)?{re.escape(left)}\s+and\s+(?:the\s+)?{re.escape(right)}\b"
                if re.search(pattern, text):
                    key = ("between", tuple(sorted((first, second))))
                    if key not in used:
                        result.append(SpatialRelation(
                            "between", (tuple(group[left]), tuple(group[right]))
                        ))
                        used.add(key)
    return result


def _range_between(candidates, start, end):
    """Distance from query XY to line segment joining two anchor centers."""
    length = end - start
    length_sq = float(np.sum(length ** 2))
    if length_sq < 1e-12:
        return np.linalg.norm(candidates - start, axis=-1), np.zeros(len(candidates))
    t = np.clip(((candidates - start) @ length) / length_sq, 0, 1)
    near = start + t[:, None] * length[None]
    return np.linalg.norm(candidates - near, axis=-1), t


def _one_anchor_scores(candidates, entity, kind, sigma_m, margin_m):
    geometry = entity["geometry_m"]
    center = polygon_center(geometry)
    if kind in DIRECTIONS:
        direction = np.asarray(DIRECTIONS[kind], np.float32)
        direction /= np.linalg.norm(direction)
        signed = (candidates - center) @ direction
        # Separate directional half-plane from the orthogonal ambiguity.
        return -np.logaddexp(0., -(signed - margin_m) / max(sigma_m, 1.))
    d = geometry_distance(candidates, geometry, is_road=entity["kind"] == "road")
    # 'near' / 'along' should prefer boundary / polyline rather than centroid.
    scale = sigma_m * (1.7 if kind == "along" else 1.)
    return -d / max(scale, 1.)


def prior_and_rule_fields(
    world_xy: np.ndarray, entities: list[dict], relations: list[SpatialRelation],
    sigma_m=25., margin_m=0.,
):
    """Return prior, rule score and per-relation feature fields [Q], [Q], [Q,R].

    *Prior* depends ONLY on referenced landmark geometry, not target labels,
    observation RGB, or direction words.
    *Rules* sum explicit named constraints, treating ambiguous IDs as OR.
    If language rules are absent, rule field is identically zero.
    """
    points = np.asarray(world_xy, dtype=np.float32).reshape(-1, 2)
    if entities:
        d = np.stack([
            geometry_distance(p, e["geometry_m"], e["kind"] == "road")
            for e in entities
        ], axis=1)
        prior = -d.min(axis=1) / max(float(sigma_m), 1.)
    else:
        prior = np.zeros(len(points), np.float32)
    feature = []
    for r in relations:
        if r.kind == "between":
            values = []
            for i in r.anchors[0]:
                for j in r.anchors[1]:
                    a = polygon_center(entities[i]["geometry_m"])
                    b = polygon_center(entities[j]["geometry_m"])
                    d_line, t = _range_between(points, a, b)
                    # Prefer inside the interval, not unbounded line.
                    values.append(-d_line / max(float(sigma_m), 1.))
            score = np.max(values, axis=0) if values else np.zeros(len(points))
        else:
            values = [
                _one_anchor_scores(points, entities[i], r.kind, sigma_m, margin_m)
                for i in r.anchors[0]
            ]
            score = np.max(values, axis=0) if values else np.zeros(len(points))
        feature.append(score.astype(np.float32))
    per_relation = (np.stack(feature, axis=-1) if feature else
                    np.zeros((len(points), 0), np.float32))
    rule_score = per_relation.sum(axis=-1) if len(relations) else np.zeros(len(points))
    return prior.astype(np.float32), rule_score.astype(np.float32), per_relation


def build_entities(
    reference_landmarks: list[dict],
    roads: list[dict],
    map_bounds: list[float],
) -> list[dict]:
    """Convert already-bound whole polygons and polylines to world meters."""
    bounds = np.asarray(map_bounds, np.float32)
    xymin, scale = bounds[:2], bounds[2:] - bounds[:2]
    if np.any(scale <= 0):
        raise ValueError("invalid bounds")
    entries = []
    for kind, group in (("building", reference_landmarks), ("road", roads)):
        for item in group:
            key = "contour" if kind == "building" else "polyline"
            geometry = np.asarray(item[key], np.float32)
            if len(geometry) < (3 if kind == "building" else 2):
                continue
            if not np.isfinite(geometry).all():
                continue
            # Source JSONL uses map-normalized coords; road GeoJSON inputs must
            # first be converted by static data loader, not silently guessed.
            entries.append({
                "kind": kind, "name": item.get("name", ""),
                "id": str(item.get("id", "")),
                "geometry_m": geometry * scale + xymin,
            })
    return entries
