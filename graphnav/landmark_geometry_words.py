"""CityNav minimal language extraction: PUBLIC landmark name + adjacent geometry phrase.

No target, appearance, colors, object categories, intermediate nodes, full
instruction BERT, GT locations, teacher poses, RGB or relation graph is ever
included in this model's input. This is a lexical ANCHOR-PHRASE extractor,
NOT verified semantic target-to-anchor scope. "parking lot behind One Stop"
is an allowed phrase "behind One Stop" but does NOT imply the car itself
is directly behind One Stop.

Only bind names that ACTUALLY OCCUR in the original instruction and match a
known public map landmark supplied by the caller. Multiple disjoint
same-name public Polygon parts are retained separately.
"""
from __future__ import annotations

import re

import numpy as np

from .geometry import resample_contour

# Store literal geometric phrase from the instruction; no rewriting it into
# east/west/positive world XY for unknown reference frames.
_PREFIX = (
    r"(?:to\s+the\s+)?(?:north|south|east|west|northeast|northwest|southeast|southwest)\s+of",
    r"(?:on\s+)?the\s+(?:north|south|east|west|left|right)\s+(?:hand\s+)?side\s+of",
    r"(?:to\s+the\s+)?(?:left|right)\s+of",
    r"(?:in\s+front\s+of|ahead\s+of|at\s+the\s+front\s+of)",
    r"(?:behind|in\s+back\s+of|at\s+the\s+rear\s+of)",
    r"(?:next\s+to|adjacent\s+to|close\s+to|near|beside)",
    r"(?:inside|within|outside(?:\s+of)?|in|on|along)",
    r"(?:opposite(?:\s+to)?|across\s+from|across\s+the\s+road\s+from)",
)
_PREFIX = re.compile(
    r"(?<!\w)(" + "|".join(_PREFIX) + r")\s+(?:the\s+)?$", re.I
)
_SUFFIX = re.compile(
    r"^\s*(?:'s|’s)\s+((?:north|south|east|west|left|right)\s+side)\b",re.I
)


def _name_occurrences(text, record):
    names=[record.get("name",""),*record.get("aliases",[])]
    found=[]
    for name in names:
        if not isinstance(name,str) or not name.strip():
            continue
        pat=re.compile(r"(?<!\w)"+re.escape(name.strip())+r"(?!\w)",re.I)
        for m in pat.finditer(text):
            found.append(dict(start=m.start(),end=m.end(),matched=text[m.start():m.end()],
                              canonical=record["name"],public_id=str(record.get("id",""))))
    return found


def extract_landmark_geometry(instruction, landmarks, *, max_anchors=16,
                              contour_vertices=16):
    """All public, visibly mentioned landmark Polygon parts, paired to
    the immediately attached spatial phrase if one exists.

    Output only name, optional literal geometry words, 16 contour XY points,
    numeric source-ID for provenance; NONE of the other instruction content.
    Invalid/overflow fragments are reported, not silently GT-selected.
    """
    if not isinstance(instruction,str) or not instruction.strip():
        raise ValueError("Nonempty public instruction required")
    if max_anchors<1 or contour_vertices<3:
        raise ValueError("Invalid extractor capacities")
    mentions=[]
    invalid=0
    for ref in landmarks:
        if not isinstance(ref,dict):
            invalid+=1;continue
        mentions.extend((m,ref) for m in _name_occurrences(instruction,ref))
    # Prefer the longest named entity when two public names overlap; never
    # match a name fragment buried in a longer landmark name.
    mentions.sort(key=lambda t:(-(t[0]["end"]-t[0]["start"]),t[0]["start"],
                                t[0]["public_id"]))
    accepted=[]
    for item,ref in mentions:
        if any(item["start"]<prev["end"] and item["end"]>prev["start"]
               and (item["start"],item["end"])!=(prev["start"],prev["end"])
               for prev,_ in accepted):
            continue
        # Exact same physical source ID repeated as an alias should not
        # duplicate a Polygon; repeated mention of the same ID is consolidated.
        if any(item["public_id"]==prev["public_id"] for prev,_ in accepted):
            continue
        accepted.append((item,ref))
    accepted.sort(key=lambda t:(t[0]["start"],t[0]["public_id"]))
    result=[]
    for m,ref in accepted:
        poly=np.asarray(ref.get("contour",[]),np.float32)
        if (poly.ndim!=2 or poly.shape[1]!=2 or len(poly)<3 or
                not np.isfinite(poly).all()):
            invalid+=1;continue
        try:
            contour=resample_contour(poly,contour_vertices)
        except ValueError:
            invalid+=1;continue
        preceding=instruction[:m["start"]]
        # Match a whole spatial phrase immediately before landmark name.
        # Do not search the whole sentence for an unrelated direction word.
        match=_PREFIX.search(preceding)
        if match:
            phrase=match.group(1)
        else:
            suffix=_SUFFIX.match(instruction[m["end"]:])
            phrase=suffix.group(1) if suffix else ""
        result.append(dict(
            name=ref["name"],geometry_words=phrase.lower(),
            contour=contour.tolist(),public_id=m["public_id"],
            phrase_present=bool(phrase),
            source="public_name_exact_text_match_plus_adjacent_lexical_geometry",
        ))
    if len(result)>max_anchors:
        raise ValueError("Public landmark mentions exceed capacity; no silent truncation")
    return dict(
        anchors=result,
        matched_names=len(result),
        paired_geometry=sum(bool(x["geometry_words"]) for x in result),
        invalid_public_contours=invalid,
        extraction_type="lexical_only_not_verified_target_direct_relation",
        target_GT_used=False,
    )
