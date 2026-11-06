"""Strictly validate an authored batch before publishing; generate no labels."""
import json
import sys
from pathlib import Path
from cache import ROOT, lines, span

DIRECTIONAL = {"behind", "front", "in_front_of", "left", "right", "left_of", "right_of",
               "above", "below", "north", "south", "east", "west", "at_back"}


def validate(path):
    source = lines(ROOT / "inputs/instructions.jsonl")
    records = json.loads(path.read_text())
    errors, indices = [], set()
    for record in records:
        index = record.get("i")
        try:
            assert index not in indices, "duplicate instruction index"
            indices.add(index)
            text = source[index]["instruction"]
            assert record.get("author_model") == "gpt-6-luna", "missing Luna provenance"
            assert record.get("author_reasoning_effort") == "low", "missing effort provenance"
            for key, entity in record["e"].items():
                assert entity[1] in {"named_landmark", "object", "region", "group"}, f"invalid entity kind {key}"
                span(text, entity[0], entity[2] if len(entity) > 2 else 0)
            for rid, relation in enumerate(record["r"]):
                subject, predicate, ref, quote, frame, confidence = relation
                refs = ref if isinstance(ref, list) else [ref]
                assert subject in record["e"], f"relation {rid}: missing subject {subject}"
                assert all(r in record["e"] for r in refs), f"relation {rid}: missing references {refs}"
                assert frame in {"N", "U", "I", "R", "W", "F"}, f"relation {rid}: invalid frame code"
                assert not (predicate in DIRECTIONAL and frame == "N"), f"relation {rid}: directional predicate using N"
                assert confidence in {"high", "medium", "low"}, f"relation {rid}: invalid confidence"
                span(text, quote)
            for rid, frame in record.get("f", {}).items():
                assert 0 <= int(rid) < len(record["r"]), "frame points outside relation array"
                if isinstance(frame.get("evidence"), str):
                    span(text, frame["evidence"])
                if frame.get("view_evidence"):
                    span(text, frame["view_evidence"])
        except Exception as exc:
            errors.append({"i": index, "error": str(exc)})
    return {"file": str(path), "records": len(records), "valid": not errors, "errors": errors}


if __name__ == "__main__":
    result = validate(Path(sys.argv[1]))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["valid"] else 1)
