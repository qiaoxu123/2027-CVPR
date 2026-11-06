"""Validate directly authored semantic records and build a lookup-only SQLite cache.

No language model, regex semantic parser, GT lookup or online fallback exists here.
"""
import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def lines(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def span(text, quote, occurrence=0):
    if not isinstance(quote, str) or not quote:
        raise ValueError("Mention/evidence must be a nonempty source substring")
    if not isinstance(occurrence, int) or occurrence < 0:
        raise ValueError("Occurrence must be a nonnegative integer")
    start = -1
    for _ in range(occurrence + 1):
        start = text.find(quote, start + 1)
        if start < 0:
            break
    if start < 0:
        raise ValueError(f"Evidence absent from instruction: {quote!r}")
    return {"text": quote, "start": start, "end": start + len(quote)}


def frame(code, rid, subject, refs, evidence):
    # These are serialization codes explicitly assigned by the assistant.
    definitions = {
        "N": ("unknown", "not_required", []),
        "U": ("unknown", "unknown", ["observer", "intrinsic", "world"]),
        "I": ("intrinsic", "unknown", ["intrinsic", "observer"]),
        "R": ("road", "multiple", ["along_road_positive", "along_road_negative"]),
        "W": ("world", "unique", ["geographic_cardinal_no_XY_calibration"]),
        "F": ("intrinsic", "multiple", ["object_front_towards_reference_region"]),
    }
    kind, status, candidates = definitions[code]
    return {"id": f"f{rid}", "type": kind, "status": status,
            "origin": subject if code == "F" else refs,
            "forward_to": refs if code == "F" else None,
            "forward_angle_world_deg": None,
            "candidate_descriptions": candidates,
            "evidence": evidence,
            "geometry_evidence": [],
            "unique_numeric_heading": False,
            "world_axes_calibrated": False}


def compile_records():
    source = lines(ROOT / "inputs/instructions.jsonl")
    records = []
    seen = set()
    for batch in sorted(ROOT.glob("direct_batch_*.json")):
        for authored in json.loads(batch.read_text()):
            if batch.name.startswith("direct_batch_luna_"):
                assert authored.get("author_model") == "gpt-6-luna", "Luna source attribution missing"
                assert authored.get("author_reasoning_effort") == "low", "Luna effort attribution missing"
            row = source[authored["i"]]
            text = row["instruction"]
            assert hashlib.sha256(text.encode()).hexdigest() == row["text_id"]
            assert row["text_id"] not in seen, "Duplicate parsing, resolve explicitly"
            seen.add(row["text_id"])
            entities = [{"id": key, "mention": span(text, value[0], value[2] if len(value) > 2 else 0), "kind": value[1],
                         "map_object_ids": [], "geometry_binding_status": "not_attempted"}
                        for key, value in authored["e"].items()]
            relations, frames = [], []
            for index, item in enumerate(authored["r"]):
                sub, predicate, ref, quote, code, confidence = item
                refs = ref if isinstance(ref, list) else [ref]
                assert sub in authored["e"] and all(r in authored["e"] for r in refs), f"{batch.name} i={authored['i']} relation={index}: undefined entity ID"
                assert confidence in {"high", "medium", "low"}
                directional = {"behind", "front", "in_front_of", "left", "right",
                               "left_of", "right_of", "above", "below", "north",
                               "south", "east", "west", "at_back"}
                assert not (predicate in directional and code == "N"), f"{batch.name} i={authored['i']} relation={index}: directional predicate cannot use frame N"
                evidence = span(text, quote)
                f = frame(code, index, sub, refs, evidence)
                override = authored.get("f", {}).get(str(index), {})
                f.update(override)
                if isinstance(f.get("evidence"), str):
                    f["evidence"] = span(text, f["evidence"])
                if "view_evidence" in f:
                    f["view_evidence"] = span(text, f["view_evidence"])
                frames.append(f)
                relations.append({"id": f"r{index}", "subject": sub,
                                  "predicate": predicate, "reference": refs,
                                  "evidence": evidence, "frame_id": f["id"],
                                  "confidence": confidence,
                                  "scope_status": "review_notes" if confidence != "high" else "explicit",
                                  "role_alternatives": authored.get("alternatives", {}).get(str(index), []),
                                  "has_named_reference": any(authored["e"][r][1] == "named_landmark" for r in refs)})
            records.append({**row, "schema_version": "1.0", "entities": entities,
                            "relations": relations, "frames": frames,
                            "uncertainties": authored["u"],
                            "provenance": {"method": "direct_current_assistant_authored",
                                           "model_requested": authored.get("author_model", "gpt-6.1-sol"),
                                           "reasoning_effort_requested": authored.get("author_reasoning_effort", "low"),
                                           "runtime_model_verified": False,
                                           "batch": batch.name,
                                           "target_GT_used": False,
                                           "map_geometry_used": False}})
    return source, records


def build():
    source, records = compile_records()
    out = ROOT / "parsed"
    out.mkdir(exist_ok=True)
    (out / "records.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    parsed = {r["text_id"] for r in records}
    pending = [r["text_id"] for r in source if r["text_id"] not in parsed]
    (out / "pending_ids.json").write_text(json.dumps(pending, indent=2))
    aliases = lines(ROOT / "inputs/aliases.jsonl")
    coverage = Counter(a["split_file"] for a in aliases if a["text_id"] in parsed)
    totals = Counter(a["split_file"] for a in aliases)
    status = {"unique_instructions": len(source), "parsed_unique": len(parsed),
              "pending_unique": len(pending), "complete": not pending,
              "relations": sum(len(r["relations"]) for r in records),
              "parsed_by_requested_model": dict(Counter(r["provenance"]["model_requested"] for r in records)),
              "split_lookup_coverage": {k: {"parsed": coverage[k], "total": totals[k]} for k in sorted(totals)},
              "protocol_sha256": hashlib.sha256((ROOT / "PROTOCOL.md").read_bytes()).hexdigest(),
              "schema_not_a_claim_of_perfect_annotation": True}
    (out / "progress.json").write_text(json.dumps(status, ensure_ascii=False, indent=2))
    with sqlite3.connect(out / "cache.sqlite") as db:
        db.execute("DROP TABLE IF EXISTS parses")
        db.execute("DROP TABLE IF EXISTS aliases")
        db.execute("CREATE TABLE parses (text_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        db.execute("CREATE TABLE aliases (split_file TEXT, episode_index INTEGER, description_index INTEGER, text_id TEXT, PRIMARY KEY(split_file, episode_index, description_index))")
        db.executemany("INSERT INTO parses VALUES (?,?)", [(r["text_id"], json.dumps(r, ensure_ascii=False)) for r in records])
        db.executemany("INSERT INTO aliases VALUES (?,?,?,?)", [(a["split_file"], a["episode_index"], a["description_index"], a["text_id"]) for a in aliases])
    return status


def lookup(*, text_id=None, split_file=None, episode_index=None, description_index=0):
    with sqlite3.connect(f"file:{ROOT / 'parsed/cache.sqlite'}?mode=ro", uri=True) as db:
        if text_id is None:
            row = db.execute("SELECT text_id FROM aliases WHERE split_file=? AND episode_index=? AND description_index=?", (split_file, episode_index, description_index)).fetchone()
            if row is None:
                raise KeyError("Unknown instruction alias")
            text_id = row[0]
        row = db.execute("SELECT payload FROM parses WHERE text_id=?", (text_id,)).fetchone()
        if row is None:
            raise KeyError(f"Parsing missing: {text_id}")
        return json.loads(row[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build", "lookup"])
    parser.add_argument("--text-id")
    parser.add_argument("--split-file")
    parser.add_argument("--episode-index", type=int)
    args = parser.parse_args()
    result = build() if args.command == "build" else lookup(text_id=args.text_id, split_file=args.split_file, episode_index=args.episode_index)
    print(json.dumps(result, ensure_ascii=False, indent=2))
