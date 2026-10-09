"""Conservative, source-type-aware CityRefer landmark name binding.

No fuzzy semantic matching, no target coordinates, and no GT-selected object.
A type suffix is removed ONLY when that suffix is compatible with the mapped
object_type (e.g., 'BMTR building' -> Building named 'BMTR').
Preserve all name collisions as separate source object IDs.
"""

from __future__ import annotations

import re


def canonical_name(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).casefold().replace("’", "'")))


TYPE_SUFFIXES = {
    "building": ("building",),
    "trafficroad": ("road", "street", "avenue", "lane"),
    "footpath": ("footpath", "path", "walkway"),
    "bridge": ("bridge",),
    "parking": ("parking lot", "car park", "parking"),
    "ground": ("ground",),
}


def type_aware_alias_match(requested: str, mapped_name: str, object_type: str) -> bool:
    requested = canonical_name(requested)
    mapped = canonical_name(mapped_name)
    if not mapped or not requested:
        return False
    if requested == mapped:
        return True
    typ = canonical_name(object_type).replace(" ", "")
    for suffix in TYPE_SUFFIXES.get(typ, ()):
        if requested == mapped + " " + suffix:
            return True
    return False


def resolve_references(reference_names, objects: dict, *, max_landmarks=16, allowed_types=None):
    """Return identity-preserving reference polygons, prioritizing exact names.

    If any exact match exists for a reference, do NOT broaden via aliases.
    Alias fallback is deterministic and checked against source object_type.
    """
    if max_landmarks < 1:
        raise ValueError("max_landmarks must be positive")
    selected, seen = [], set()
    sorted_objects = sorted(objects.items(), key=lambda pair: str(pair[0]))
    for requested in reference_names:
        key = canonical_name(requested)
        if not key:
            continue
        acceptable = []
        for object_id, obj in sorted_objects:
            typ = str(obj.get("object_type", ""))
            if allowed_types is not None and typ not in allowed_types:
                continue
            name = str(obj.get("name", ""))
            contour = obj.get("contour", [])
            if not name or len(contour) < 3:
                continue
            if type_aware_alias_match(requested, name, typ):
                acceptable.append((str(object_id), name, typ, contour))
        exact = [m for m in acceptable if canonical_name(m[1]) == key]
        candidates = exact if exact else acceptable
        for object_id, name, typ, contour in candidates:
            if object_id in seen:
                # Keep all phrases previously matched to the same source ID.
                for item in selected:
                    if item["id"] == object_id and requested not in item["aliases"]:
                        item["aliases"].append(str(requested))
                        break
                continue
            selected.append({
                "id": object_id,
                "name": name,
                "object_type": typ or "Unknown",
                "aliases": [str(requested)] if canonical_name(name) != key else [],
                "contour": contour,
            })
            seen.add(object_id)
            if len(selected) >= max_landmarks:
                return selected
    return selected
