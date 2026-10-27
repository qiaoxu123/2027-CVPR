"""Conservative, efficient CityRefer named polygon binding.

A type suffix is removed only for a compatible source object_type.
Whole polygons stay distinct by their unique map object ID. Exact name
matches outrank typed alias matches, and neither ever uses GT positions.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher


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
    allowed_types=None, index=None, match_mode="typed",
    fuzzy_threshold=0.88, fuzzy_margin=0.04,
):
    """Return every exact or typed-suffix matching polygon ID without GT lookup.

    If an exact match exists, typed suffix fallback is disabled for that query.
    Homonymous IDs survive independently up to the transparent max_landmarks cap.
    """
    if max_landmarks < 1:
        raise ValueError("max_landmarks must be positive")
    if match_mode not in ("exact", "typed", "fuzzy"):
        raise ValueError("match_mode must be exact|typed|fuzzy")
    if not 0 < fuzzy_threshold <= 1:
        raise ValueError("invalid fuzzy_threshold")
    index = build_name_index(objects) if index is None else index
    selected, seen = [], {}
    for requested in reference_names:
        key = canonical_name(requested)
        if not key:
            continue
        exact = [entry for entry in index["exact"].get(key, [])
                 if allowed_types is None or entry[2] in allowed_types]
        typed = [] if match_mode == "exact" else [
            entry for entry in index["typed"].get(key, [])
            if allowed_types is None or entry[2] in allowed_types
        ]
        matches = exact if exact else typed
        if not matches and match_mode == "fuzzy":
            # Fuzzy matching is ONLY a fallback to typed exact binding.
            # Match explicit suffix types conservatively: road != building.
            allowed = None
            for typ, suffixes in TYPE_SUFFIXES.items():
                if any(key.endswith(" " + suffix) for suffix in suffixes):
                    allowed = typ
                    break
            rankings = []
            for candidate_name, entries in index["exact"].items():
                for entry in entries:
                    source_type = canonical_name(entry[2]).replace(" ", "")
                    if allowed is not None and source_type != allowed:
                        continue
                    if allowed_types is not None and entry[2] not in allowed_types:
                        continue
                    ratio = SequenceMatcher(None, key, candidate_name).ratio()
                    # 'Main Road' can match 'Main Rd' without changing type,
                    # but it must NOT indiscriminately match an unrelated object.
                    for suffix in TYPE_SUFFIXES.get(source_type, ()):
                        ratio = max(
                            ratio,
                            SequenceMatcher(None, key, candidate_name + " " + suffix).ratio(),
                        )
                    rankings.append((ratio, candidate_name, entry))
            if rankings:
                rankings.sort(key=lambda x: (-x[0], x[1], x[2][0]))
                best_score, best_name, _ = rankings[0]
                competitor = max(
                    (score for score, name, _ in rankings if name != best_name),
                    default=0.0,
                )
                if best_score >= fuzzy_threshold and best_score - competitor >= fuzzy_margin:
                    matches = [entry for _, name, entry in rankings if name == best_name]
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
