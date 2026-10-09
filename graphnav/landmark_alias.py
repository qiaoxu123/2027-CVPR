"""Conservative, efficient CityRefer named polygon binding.

A type suffix is removed only for a compatible source object_type.
Whole polygons stay distinct by their unique map object ID. Exact name
matches outrank typed alias matches, and neither ever uses GT positions.
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
    req, name = canonical_name(requested), canonical_name(mapped_name)
    if not req or not name:
        return False
    if req == name:
        return True
    type_key = canonical_name(object_type).replace(" ", "")
    return any(req == name + " " + suffix
               for suffix in TYPE_SUFFIXES.get(type_key, ()))


def build_name_index(objects: dict) -> dict:
    """Build once per map; O(1) lookup per referenced name, no fuzzy matching."""
    exact, typed = {}, {}
    for object_id, obj in sorted(objects.items(), key=lambda pair: str(pair[0])):
        name = str(obj.get("name", ""))
        contour = obj.get("contour", [])
        if not name or len(contour) < 3:
            continue
        key = canonical_name(name)
        if not key:
            continue
        item = (str(object_id), name, str(obj.get("object_type", "Unknown")), contour)
        exact.setdefault(key, []).append(item)
        source_type = canonical_name(item[2]).replace(" ", "")
        for suffix in TYPE_SUFFIXES.get(source_type, ()):
            typed.setdefault(key + " " + suffix, []).append(item)
    return {"exact": exact, "typed": typed}


def resolve_references(
    reference_names, objects: dict, *, max_landmarks=16,
    allowed_types=None, index=None,
):
    """Return every exact or typed-suffix matching polygon ID without GT lookup.

    If an exact match exists, typed suffix fallback is disabled for that query.
    Homonymous IDs survive independently up to the transparent max_landmarks cap.
    """
    if max_landmarks < 1:
        raise ValueError("max_landmarks must be positive")
    index = build_name_index(objects) if index is None else index
    selected, seen = [], {}
    for requested in reference_names:
        key = canonical_name(requested)
        if not key:
            continue
        exact = [entry for entry in index["exact"].get(key, [])
                 if allowed_types is None or entry[2] in allowed_types]
        matches = exact if exact else [
            entry for entry in index["typed"].get(key, [])
            if allowed_types is None or entry[2] in allowed_types
        ]
        for object_id, name, source_type, contour in matches:
            if object_id in seen:
                found = selected[seen[object_id]]
                if key != canonical_name(name) and requested not in found["aliases"]:
                    found["aliases"].append(str(requested))
                continue
            selected.append({
                "id": object_id,
                "name": name,
                "object_type": source_type,
                "aliases": [str(requested)] if key != canonical_name(name) else [],
                "contour": contour,
            })
            seen[object_id] = len(selected) - 1
            if len(selected) >= max_landmarks:
                return selected
    return selected
