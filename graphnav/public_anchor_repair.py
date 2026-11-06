"""Controlled PUBLIC-ONLY anchor binding repair for the frozen Unified V2.

This is a TEST intervention, not a hidden fix to the deployed baseline.

- Keep original full instruction and public Polygon identity.
- Do not inspect target ID/GT, endpoint distance, oracle yaw or trajectory.
- Never fabricate a missing place name. No fuzzy guesses are activated.
- Only recover invalid contours when a unique simple Polygon is restored
  with a closely matching area and bounding box.
- For >16 Polygon references, keep complete *named mention groups* in
  instruction order when they fit; no GT-nearest or partial homonym subset.
- If one single group exceeds the frozen 16-node capacity, it stays
  unsupported. Audit explicitly; do not misreport the task as repaired.
"""
from __future__ import annotations

from collections import Counter
import math

import numpy as np
import shapely
from shapely.geometry import Polygon

from graphnav.geometry import normalize_xy
from scripts.prepare_unified_v2_landmark_relation_ablation import (
    public_references, extract_pairs, normalized_instruction
)


def _simple_polygon(candidate):
    return (candidate.geom_type == "Polygon" and not candidate.is_empty
            and candidate.is_valid and candidate.area > 0
            and len(candidate.interiors) == 0)


def conservative_repair_contour(values, *, max_area_change=0.03,
                                max_bbox_shift_fraction=0.005):
    """Return repaired contour or None; reject split/filled/unknown shapes.

    A self-intersecting boundary that becomes MultiPolygon is not safely
    converted into one fake outline. No target position is used.
    """
    try:
        raw = np.asarray(values, dtype=np.float64)
    except (TypeError,ValueError):
        return None
    if (raw.ndim != 2 or raw.shape[1] != 2 or len(raw) < 3
            or not np.isfinite(raw).all()):
        return None
    source = Polygon(raw)
    if _simple_polygon(source):
        return raw.tolist()
    try:
        from shapely.validation import make_valid
        repaired = make_valid(source)
    except (ImportError, AttributeError):
        return None
    if not _simple_polygon(repaired):
        return None
    area = abs(float(source.area))
    if area <= 1e-9 or abs(repaired.area-area)/area > max_area_change:
        return None
    old = np.asarray(source.bounds)
    new = np.asarray(repaired.bounds)
    size = max(np.ptp(raw[:,0]), np.ptp(raw[:,1]), 1.0)
    if np.max(np.abs(new-old)) > max_bbox_shift_fraction*size:
        return None
    output=np.asarray(repaired.exterior.coords, dtype=np.float64)
    if len(output)>3 and np.allclose(output[0],output[-1]):
        output=output[:-1]
    if len(output)<3 or not np.isfinite(output).all():
        return None
    return output.tolist()


def _public_catalog_with_safe_repairs(map_objects):
    """Per-episode private view; public source JSON remains immutable."""
    edited=dict(map_objects)
    audited=[]
    for oid,item in map_objects.items():
        try:
            polygon=np.asarray(item.get("contour",[]),dtype=np.float64)
        except (TypeError,ValueError):
            polygon=np.empty((0,2),dtype=np.float64)
        legal=(polygon.ndim==2 and polygon.shape[1:]==(2,)
               and len(polygon)>=3 and np.isfinite(polygon).all())
        if legal:
            poly=Polygon(polygon)
            legal=_simple_polygon(poly)
        if legal:continue
        repaired=conservative_repair_contour(item.get("contour",[]))
        if repaired is not None:
            edited[oid]={**item,"contour":repaired}
            audited.append(str(oid))
    return edited,audited


def select_complete_groups(mentions, candidates, capacity):
    """Keep complete overlapping public-ID components, no partial names.

    A single physical Polygon may have multiple aliases. Mention ID groups
    that overlap are therefore merged BEFORE any capacity decision; this
    prevents accidentally including half a homonymous group when one name
    was a subset of another alias query.
    """
    present={str(r["id"]) for r in candidates}
    groups=[]
    for mention in mentions:
        ids=set(str(x) for x in mention["public_ids"] if str(x) in present)
        if not ids:continue
        merged=dict(ids=ids,names=[mention["text"]])
        overlapping=[g for g in groups if g["ids"] & merged["ids"]]
        for g in overlapping:
            merged["ids"].update(g["ids"])
            merged["names"].extend(g["names"])
            groups.remove(g)
        groups.append(merged)
    # If late alias connections created transitive overlaps, merge again.
    changed=True
    while changed:
        changed=False
        for i in range(len(groups)):
            for j in range(i+1,len(groups)):
                if groups[i]["ids"] & groups[j]["ids"]:
                    groups[i]["ids"].update(groups[j]["ids"])
                    groups[i]["names"].extend(groups[j]["names"])
                    groups.pop(j)
                    changed=True
                    break
            if changed:break
    kept=set();events=[]
    for group in groups:
        ids=group["ids"]
        if len(ids)>capacity:
            status="single_group_over_capacity"
        elif len(kept)+len(ids-kept)>capacity:
            status="no_capacity_for_entire_group"
        else:
            status="complete_group_kept"
            kept.update(ids)
        events.append(dict(names=group["names"],reason=status,
                           matching_public_ids=len(ids),
                           selected=len(ids) if status=="complete_group_kept" else 0))
    return [ref for ref in candidates if str(ref["id"]) in kept],events


def sanitize_public_map(map_objects):
    """Public-only, immutable map copy; may be cached by the online Policy."""
    return _public_catalog_with_safe_repairs(map_objects)


def repaired_public_text_record(map_name,instruction,bounds,catalog,matcher,*,
                                mode="full_text",max_landmarks=16,
                                repair_geometry=True,
                                repair_overflow=True,
                                preverified_repaired_ids=()):
    """Pure public-only alternative to the existing public_text_record.

    Returns unchanged public record format and an explicit intervention audit.
    """
    if mode not in ("full_text","landmark_relation"):
        raise ValueError("Only original public text-binding arms supported")
    if max_landmarks<1:
        raise ValueError("Invalid frozen model capacity")
    mentions=matcher.match(map_name,instruction)
    pairs=extract_pairs(instruction,mentions)
    public_map=catalog[map_name]
    safe_catalog=dict(catalog)
    if repair_geometry:
        safe_map,repaired_ids=_public_catalog_with_safe_repairs(public_map)
        safe_catalog[map_name]=safe_map
    else:
        repaired_ids=list(preverified_repaired_ids)
    # Unlimited at extraction time, not at model input time. The reference
    # list is always bounded below, preserving the original network shape.
    all_refs, geometry_warnings=public_references(
        map_name,bounds,mentions,safe_catalog,max_landmarks=1000000)
    if len(all_refs)<=max_landmarks:
        refs=all_refs
        groups=[]
    elif repair_overflow:
        refs,groups=select_complete_groups(mentions,all_refs,max_landmarks)
    else:
        refs=[]
        groups=[dict(reason="old_overflow_all_empty",
                     matching_public_ids=len(all_refs),selected=0)]
    if len(refs)>max_landmarks:
        raise AssertionError("Frozen model max_landmarks violated")
    kept_ids={str(r["id"]) for r in refs}
    retained=[p for p in pairs if any(str(i) in kept_ids for i in p["public_ids"])]
    text=instruction if mode=="full_text" else normalized_instruction(retained,mode)
    record=dict(map_name=map_name,instruction=text,world_bounds=list(bounds),
                physical_map_sizes=[bounds[2]-bounds[0],bounds[3]-bounds[1]],
                landmarks=refs,dataset_version="revised",
                match_mode="fuzzy",target_xy=[0.,0.])
    outcome=dict(
        original_instruction=instruction,mode=mode,
        public_name_mentions=len(mentions),
        public_polygon_legal_before_capacity=len(all_refs),
        public_polygon_selected=len(refs),
        polygon_ids=[r["id"] for r in refs],
        repaired_public_ids=[oid for oid in repaired_ids if oid in {
            str(i) for m in mentions for i in m["public_ids"]}],
        complete_group_events=groups,
        truncated_group_count=sum(g["reason"]!="complete_group_kept" for g in groups),
        geometry_warnings=geometry_warnings,
        still_unbound=not bool(refs),
        GT_used_to_bind=False,
        distance_used_to_select=False,
        name_fuzzy_guess_used=False,
    )
    return record,outcome
