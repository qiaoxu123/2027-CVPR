"""Explicit conservative cardinal phrase swaps for causal language-head diagnostics.

Only the UNIQUE pattern 'north/south/east/west of' is modified, and never
inside an identified proper public landmark name. The base Geometry+Relation
output remains identical between two passes to isolate the new language head.
This is a robustness probe, NOT a validated ground-truth relation annotation.
"""
import re


_CARDINAL = {
    "north":"south","south":"north","east":"west","west":"east",
}
_PATTERN = re.compile(r"\b(north|south|east|west)\s+of\b",re.I)


def swap_single_cardinal(text, public_names=()):
    hits=list(_PATTERN.finditer(text))
    if len(hits)!=1:
        return None
    hit=hits[0]
    for name in public_names:
        cleaned=name.strip()
        if not cleaned:
            continue
        # Static map names can be prefixed with the source category.
        for prefix in ("Building ","TrafficRoad ","Footpath ","Bridge ",
                       "Parking ","Ground ","Road "):
            if cleaned.startswith(prefix):
                cleaned=cleaned[len(prefix):]
                break
        for label_hit in re.finditer(re.escape(cleaned),text,re.I):
            if hit.start()<label_hit.end() and hit.end()>label_hit.start():
                return None
    old=hit.group(1).lower()
    new=_CARDINAL[old]
    mutated=text[:hit.start(1)]+new+text[hit.end(1):]
    return dict(original=text,mutated=mutated,old=old,new=new,
                phrase_span=[hit.start(),hit.end()],
                note="world direction not verified by text alone")
