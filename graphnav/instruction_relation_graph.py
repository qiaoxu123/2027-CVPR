"""Conservative navigation instruction -> typed TARGET/REFERENCE relation graph.

Adapter for graphnav.structured_navigation.parse_instruction(...) output.
This is a *graph representation*, NOT an independently validated parser:
the upstream parser emits proposals, and all uncertainties are retained.

Geometry is derived exclusively from PUBLIC map Polygons and world bounds.
A target or intermediate object without a public location has missing geometry;
we NEVER assign a target GT coordinate or silently collapse indirect relations.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np

ROLES = ("target", "intermediate", "public_landmark", "reference_object")
PREDICATES = (
    "unknown", "inside", "outside", "on", "near", "next_to", "along",
    "left_of", "right_of", "in_front_of", "behind",
    "north_of", "south_of", "east_of", "west_of",
    "between", "across_from", "surrounded_by", "part_of",
)
FRAMES = ("unknown", "topological", "metric_or_adjacency",
          "reference_frame_unknown", "world_cardinal")
NODE_GEO_DIM = 8
EDGE_GEO_DIM = 9

_PATTERNS = [
    ("in_front_of", r"\b(?:in\s+front\s+of|ahead\s+of)\b"),
    ("across_from", r"\b(?:across\s+(?:the\s+street\s+)?from|opposite)\b"),
    ("next_to", r"\b(?:next\s+to|adjacent\s+to|beside)\b"),
    ("north_of", r"\bnorth(?:ern)?\s+(?:side\s+)?of\b|\bnorth\b"),
    ("south_of", r"\bsouth(?:ern)?\s+(?:side\s+)?of\b|\bsouth\b"),
    ("east_of", r"\beast(?:ern)?\s+(?:side\s+)?of\b|\beast\b"),
    ("west_of", r"\bwest(?:ern)?\s+(?:side\s+)?of\b|\bwest\b"),
    ("left_of", r"\bleft(?:\s+(?:hand\s+)?side)?(?:\s+of)?\b"),
    ("right_of", r"\bright(?:\s+(?:hand\s+)?side)?(?:\s+of)?\b"),
    ("behind", r"\b(?:behind|at\s+the\s+back\s+of|rear\s+of)\b"),
    ("inside", r"\b(?:inside|within|in)\b"),
    ("outside", r"\boutside\b"),
    ("surrounded_by", r"\b(?:surrounded|enclosed)\s+by\b"),
    ("between", r"\bbetween\b"),
    ("along", r"\balong\b"),
    ("near", r"\b(?:near|close\s+to)\b"),
    ("on", r"\bon\b"),
    ("part_of", r"\b(?:possessive\s+region\s+association|associated\s+region|of|with)\b"),
]
_PATTERNS = [(name, re.compile(pattern, re.I)) for name, pattern in _PATTERNS]
_WORLD_DIRS = {
    "north_of": (0., 1.), "south_of": (0., -1.),
    "east_of": (1., 0.), "west_of": (-1., 0.),
}


def normalize_predicate(phrase: str) -> str:
    """Semantic taxonomy only; NOT claim that relation scope is resolved."""
    for name, pat in _PATTERNS:
        if pat.search(phrase or ""):
            return name
    return "unknown"


def _bounds(bounds: Any) -> tuple[np.ndarray, np.ndarray]:
    b = np.asarray(bounds, dtype=np.float64)
    if b.shape != (4,) or not np.isfinite(b).all() or (b[2:] <= b[:2]).any():
        raise ValueError("Valid world bounds in meters required")
    return b[:2], b[2:]-b[:2]


def _public_geometry(identifiers, public_objects, world_bounds):
    """Normalized bbox proxy only; Polygon itself remains separate and public."""
    lo, span = _bounds(world_bounds)
    segments = []
    for identifier in identifiers:
        entry = public_objects.get(str(identifier))
        if entry is None:
            continue
        xy = np.asarray(entry.get("contour", []), dtype=np.float64)
        if (xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3 or
                not np.isfinite(xy).all()):
            continue
        try:
            import shapely.geometry
            region = shapely.geometry.Polygon(xy)
            if not region.is_valid or region.area <= 0:
                continue
        except ImportError:
            # Fail safe when Polygon validation is unavailable.
            continue
        segments.append(region)
    if not segments:
        return [0.] * NODE_GEO_DIM, False
    import shapely.ops
    geometry = shapely.ops.unary_union(segments)
    low = np.array(geometry.bounds[:2], dtype=np.float64)
    high = np.array(geometry.bounds[2:], dtype=np.float64)
    center = (low + high)/2.
    wh = high - low
    area = float(geometry.area)  # no double-count for overlapping public parts
    v = [
        *((center-lo)/span), *(wh/span),
        np.log1p(max(area, 0.))/np.log1p(max(float(np.prod(span)), 1.)),
        float(len(segments))/16.,
        float(wh[0]/max(wh[1], 1e-6)),
        1.,
    ]
    return [float(x) for x in v], True


def _edge_geo(src_geo, dst_geo, src_ok: bool, dst_ok: bool,
              bounds) -> tuple[list[float], bool]:
    """Pair displacement is valid only when BOTH entities have public geometry."""
    if not (src_ok and dst_ok):
        return [0.] * EDGE_GEO_DIM, False
    _, span = _bounds(bounds)
    dx = np.asarray(dst_geo[:2]) - np.asarray(src_geo[:2])
    xy = dx * span
    distance = np.linalg.norm(xy)
    unit = xy/max(distance, 1e-6)
    out = [float(dx[0]), float(dx[1]), float(distance/np.linalg.norm(span)),
           float(unit[0]), float(unit[1]),
           float(src_geo[2]), float(src_geo[3]),
           float(dst_geo[2]), float(dst_geo[3])]
    return out, True


def compile_relation_graph(parsed: dict, public_objects: dict,
                           world_bounds, *, world_north_verified: bool = False) -> dict:
    """Convert existing *unverified* text parse to a typed, auditable graph.

    Supplied public_objects must be the map-specific {source_id: metadata} dict.
    parsed MUST be from the public text/catalog parser, not an annotation label.
    """
    if parsed.get("target_GT_used") is not False:
        raise ValueError("Unverified GT-free parser provenance")
    forbidden = ("target_world", "target_xy", "target_position", "teacher_poses")
    if any(k in parsed for k in forbidden):
        raise ValueError("Private oracle labels must not enter graph compiler")
    _bounds(world_bounds)
    original = parsed["original_text"]
    target = (parsed.get("target") or {}).get("entity")
    raw_entities = parsed.get("entities", [])
    if len({x["id"] for x in raw_entities}) != len(raw_entities):
        raise ValueError("Duplicate entity ID")
    raw_lookup = {x["id"]:x for x in raw_entities}
    mentions = parsed.get("landmarks", [])
    nodes = []
    for item in raw_entities:
        a,b = item["span"]
        if not (0 <= a <= b <= len(original)):
            raise ValueError("Invalid phrase span")
        overlapping = [m for m in mentions
                       if a < m.get("end", -1) and b > m.get("start", 10**9)]
        ids = []
        for m in overlapping:
            ids.extend(str(oid) for oid in m.get("public_ids", []))
        ids = list(dict.fromkeys(ids))
        geo, geovalid = _public_geometry(ids, public_objects, world_bounds)
        role = ("target" if item["id"] == target
                else "public_landmark" if ids or item.get("role")=="landmark"
                else "intermediate" if item.get("role")=="reference_object"
                else "reference_object")
        target_info = parsed.get("target") or {}
        attrs = list(item.get("attributes", []))
        if item["id"] == target:
            attrs += list(target_info.get("attributes", []))
        nodes.append(dict(
            id=item["id"], role=role, phrase=item["text"], span=[int(a),int(b)],
            category=target_info.get("category") if item["id"] == target else None,
            attributes=attrs,
            public_names=item.get("public_names", []),
            public_ids=ids, geometry=geo, geometry_valid=geovalid,
        ))
    node_by_id = {x["id"]:x for x in nodes}
    if target is not None and target not in node_by_id:
        raise ValueError("Target ID not found in node set")
    edges = []
    for item in parsed.get("relations", []):
        if item["subject"] not in node_by_id or item["object"] not in node_by_id:
            raise ValueError("Relation references missing entity")
        kind = normalize_predicate(item.get("predicate", ""))
        src = node_by_id[item["subject"]]; dst = node_by_id[item["object"]]
        geo, geo_valid = _edge_geo(
            src["geometry"], dst["geometry"],
            src["geometry_valid"], dst["geometry_valid"], world_bounds
        )
        frame = item.get("reference_frame", {}).get("frame", "unknown")
        if frame not in FRAMES:
            frame = "unknown"
        ambiguous = list(item.get("ambiguity", []))
        if frame == "reference_frame_unknown":
            ambiguous.append("reference_frame_unresolved")
        if frame == "world_cardinal" and not world_north_verified:
            ambiguous.append("geographic_north_not_verified")
        vector = [0.,0.]
        dir_valid = False
        if frame == "world_cardinal" and world_north_verified and kind in _WORLD_DIRS:
            vector = list(_WORLD_DIRS[kind]);dir_valid = True
        edges.append(dict(
            id=item["id"], subject=item["subject"], object=item["object"],
            predicate=kind, source_predicate=item.get("predicate", ""),
            source_text=item.get("source_text", ""),
            source_span=item.get("source_span", None),
            relation_kind=item.get("relation_kind", "unknown"),
            frame=frame, ambiguity=sorted(set(ambiguous)),
            direction_xy=vector, direction_xy_valid=dir_valid,
            geometry=geo, geometry_valid=geo_valid,
        ))
    edge_by_id = {x["id"]:x for x in edges}
    paths = []
    for item in parsed.get("direct_relations", [])+parsed.get("indirect_relations", []):
        chain=item.get("relations", [])
        node_ids=item.get("entities", [])
        if not node_ids or len(node_ids) != len(chain)+1 or node_ids[0] != target:
            raise ValueError("Malformed target-anchored relation path")
        if any(x not in node_by_id for x in node_ids):
            raise ValueError("Path references missing node")
        path_edges = []
        issues = list(item.get("ambiguity", []))
        for i, step in enumerate(chain):
            edge=edge_by_id.get(step["relation_id"])
            if edge is None:
                raise ValueError("Path references missing edge")
            travel = step["traversal"]
            if travel not in ("forward","reverse"):
                raise ValueError("Invalid relation traversal")
            s, o = node_ids[i:i+2]
            if ((travel=="forward" and (edge["subject"],edge["object"])!=(s,o)) or
                    (travel=="reverse" and (edge["object"],edge["subject"])!=(s,o))):
                raise ValueError("Path endpoints/traversal inconsistent")
            if travel=="reverse":
                issues.append("reverse_relation_requires_semantic_inverse")
            issues.extend(edge["ambiguity"])
            if edge["predicate"]=="unknown" or edge["relation_kind"]!="spatial_proposal":
                issues.append("predicate_or_relation_scope_unverified")
            path_edges.append(dict(id=edge["id"], traversal=travel))
        path_end=node_by_id[node_ids[-1]]
        if not path_end["public_ids"]:
            issues.append("terminal_landmark_unbound")
        paths.append(dict(
            kind="direct" if len(chain)==1 else "indirect",
            hops=len(chain), nodes=list(node_ids), edges=path_edges,
            landmark_names=item.get("public_names", []),
            ambiguity=sorted(set(issues)),
            eligible_for_geometric_scoring=not bool(issues),
            # Strictly diagnostic: no input can become verified oracle target.
            parser_proposal_not_ground_truth=True,
        ))
    return dict(
        schema="citynav-instruction-relation-graph-v1",
        instruction=original, map_name=parsed.get("map_name"),
        target_id=target, nodes=nodes, edges=edges, paths=paths,
        global_ambiguity=list(parsed.get("ambiguity", [])),
        target_GT_used=False, neural_parser_trained=False,
        world_north_verified=bool(world_north_verified),
        summaries=dict(direct_paths=sum(p["kind"]=="direct" for p in paths),
                       indirect_paths=sum(p["kind"]=="indirect" for p in paths),
                       unresolved_paths=sum(bool(p["ambiguity"]) for p in paths),
                       ungrounded_nodes=sum(not n["geometry_valid"] for n in nodes)),
    )


def tensorize_relation_graph(graph: dict, *, max_nodes=32, max_edges=64) -> dict:
    """Fixed-size learned-encoder input. Never truncate a graph silently."""
    if len(graph["nodes"])>max_nodes or len(graph["edges"])>max_edges:
        raise ValueError("Graph too large: increase capacity, do not truncate")
    nodes = np.zeros((max_nodes,NODE_GEO_DIM), np.float32)
    roles = np.zeros(max_nodes,np.int64)
    node_mask=np.zeros(max_nodes,bool);node_geo_mask=np.zeros(max_nodes,bool)
    edge_index=np.zeros((max_edges,2),np.int64)
    predicates=np.zeros(max_edges,np.int64)
    frames=np.zeros(max_edges,np.int64)
    edge_geometry=np.zeros((max_edges,EDGE_GEO_DIM),np.float32)
    edge_mask=np.zeros(max_edges,bool);edge_geo_mask=np.zeros(max_edges,bool)
    edge_direction=np.zeros((max_edges,2),np.float32)
    edge_direction_mask=np.zeros(max_edges,bool)
    ids={x["id"]:i for i,x in enumerate(graph["nodes"])}
    for i,x in enumerate(graph["nodes"]):
        nodes[i]=x["geometry"];roles[i]=ROLES.index(x["role"])
        node_mask[i]=True;node_geo_mask[i]=x["geometry_valid"]
    for i,x in enumerate(graph["edges"]):
        edge_index[i]=ids[x["subject"]],ids[x["object"]]
        predicates[i]=PREDICATES.index(x["predicate"])
        frames[i]=FRAMES.index(x["frame"])
        edge_geometry[i]=x["geometry"];edge_mask[i]=True
        edge_geo_mask[i]=x["geometry_valid"]
        edge_direction[i]=x["direction_xy"]
        edge_direction_mask[i]=x["direction_xy_valid"]
    return dict(
        node_geometry=nodes, node_roles=roles, node_mask=node_mask,
        node_geo_mask=node_geo_mask, edge_index=edge_index,
        predicates=predicates, frames=frames,
        edge_geometry=edge_geometry, edge_mask=edge_mask,
        edge_geo_mask=edge_geo_mask, edge_direction=edge_direction,
        edge_direction_mask=edge_direction_mask,
        node_phrases=[x["phrase"] for x in graph["nodes"]],
        edge_phrases=[x["source_predicate"] for x in graph["edges"]],
        target_index=ids.get(graph["target_id"],-1),
    )
