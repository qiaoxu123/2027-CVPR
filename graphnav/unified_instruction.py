"""Public-text-only instruction adapter for UnifiedPGVB.

A complete named-landmark command is supported for inference. Direct and
indirect target relation commands are parsed as diagnostics only: target
appearance heads and language reference frames are not yet validated.
No GT target coordinates, instances or teacher trajectories are inputs.
"""
import re

from .landmark_alias import build_name_index, canonical_name

VERSION = "unified-instruction-v1"
_NAMED = re.compile(
    r"^\s*(?:navigate|go|fly|head|travel)\s+to\s+(?P<name>.+?)\s*[.!]?\s*$",
    re.IGNORECASE,
)


def _unresolved(text, map_name, reason):
    return dict(
        version=VERSION, original_instruction=text, map_name=map_name,
        intent="unresolved", approved_for_inference=False, reason=reason,
        target=None, landmark=None, landmarks=[], relations=[],
        ambiguity=[reason], target_GT_used=False,
    )


def _public_landmark(requested, objects):
    """Only exact/typed public-name binding; retain all same-name polygon parts."""
    index = build_name_index(objects)
    key = canonical_name(requested)
    matches = index["exact"].get(key) or index["typed"].get(key) or []
    if not matches:
        return None, "name_not_in_public_catalog"
    if len({canonical_name(item[1]) for item in matches}) != 1:
        return None, "ambiguous_public_name"
    kinds = {item[2] for item in matches}
    if kinds == {"Building"}:
        kind = "Building"
    elif kinds <= {"TrafficRoad", "Footpath", "Bridge"}:
        kind = "Road"
    else:
        return None, "unsupported_or_mixed_public_types"
    return dict(
        name=matches[0][1], type=kind,
        public_ids=[item[0] for item in matches],
        requested_text=requested, binding="exact_public_catalog",
    ), None


def compile_structured_instruction(parsed):
    """Adapt existing structured_navigation parser output conservatively.

    Preserve original relation scope, traversal and ambiguity. Parsing does not
    validate physical meaning or justify using a target inference head.
    """
    if parsed.get("target_GT_used") is not False:
        raise ValueError("Structured input must explicitly be GT-free")
    text, map_name = parsed["original_text"], parsed["map_name"]
    mentions = [
        dict(names=m.get("public_names", []),
             ids=[str(x) for x in m.get("public_ids", [])],
             types=m.get("types", []),
             span=[m["start"], m["end"]])
        for m in parsed.get("landmarks", [])
    ]
    paths = parsed.get("direct_relations", []) + parsed.get("indirect_relations", [])
    if not paths:
        out = _unresolved(text, map_name, "no_reliable_target_landmark_path")
        out["landmarks"] = mentions
        out["target"] = parsed.get("target")
        out["ambiguity"] = sorted(set(out["ambiguity"] + parsed.get("ambiguity", [])))
        return out
    edges = {r["id"]: r for r in parsed.get("relations", [])}
    relations = []
    for path in paths:
        chain = []
        for ref in path.get("relations", []):
            edge = edges.get(ref.get("relation_id"))
            if edge is None:
                raise ValueError("Relation path references missing edge")
            chain.append(dict(
                predicate=edge["predicate"], traversal=ref["traversal"],
                source_span=edge["source_span"],
                frame=edge["reference_frame"]["frame"],
                scope=edge["relation_kind"],
                ambiguity=list(edge.get("ambiguity", [])),
            ))
        relations.append(dict(
            hops=path["hops"], landmark_names=path.get("landmark_names", []),
            chain=chain, ambiguity=list(path.get("ambiguity", [])),
        ))
    direct = [r for r in relations if r["hops"] == 1]
    indirect = [r for r in relations if r["hops"] > 1]
    intent = (
        "direct_target" if len(direct) == 1 and not indirect
        else "indirect_target" if indirect and not direct
        else "ambiguous_target_relations"
    )
    ambiguity = list(parsed.get("ambiguity", []))
    if any(e["frame"] == "world_cardinal" for p in relations for e in p["chain"]):
        ambiguity.append("geographic_north_not_independently_verified")
    return dict(
        version=VERSION, original_instruction=text, map_name=map_name,
        intent=intent, approved_for_inference=False,
        reason="target_visual_head_not_trained_or_relation_unverified",
        target=parsed.get("target"), landmark=None, landmarks=mentions,
        relations=relations, ambiguity=sorted(set(ambiguity)),
        target_GT_used=False, parser_status=parsed.get("parser_status"),
    )


def parse_unified_instruction(
    text, map_name, public_objects, *, nlp=None, public_names=None,
):
    """Online named-goal fast path; optional existing spaCy parser for diagnostics.

    Unrecognized/complex instructions never silently become a public-landmark
    navigation goal. The online named-goal path requires no spaCy or new LLM.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Nonempty original instruction required")
    match = _NAMED.fullmatch(text)
    if match:
        phrase = match.group("name").strip().rstrip(".!").strip()
        landmark, error = _public_landmark(phrase, public_objects)
        if error:
            return _unresolved(text, map_name, error)
        return dict(
            version=VERSION, original_instruction=text, map_name=map_name,
            intent="navigate_to_landmark", approved_for_inference=True,
            reason="explicit_named_goal",
            target=dict(
                name=landmark["name"], category=landmark["type"],
                attributes=[], same_as_landmark=True,
            ),
            landmark=landmark, landmarks=[landmark], relations=[],
            ambiguity=[],
            source_span=[match.start("name"), match.end("name")],
            target_GT_used=False,
        )
    if nlp is None or public_names is None:
        return _unresolved(text, map_name, "complex_instruction_needs_parser")
    from .structured_navigation import parse_instruction
    parsed = parse_instruction(text, map_name, nlp, public_names)
    return compile_structured_instruction(parsed)
