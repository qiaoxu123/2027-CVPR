"""Inspect UnifiedPGVB language parsing from public CityNav text only.

Input JSONL rows: map_name, instruction (optional episode/instruction_index).
No private target IDs, coordinates, future RGB or teacher information accepted.
Rich parser mode uses existing structured_navigation and spaCy proposals.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from graphnav.unified_instruction import parse_unified_instruction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rich", action="store_true")
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    public_catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    nlp, names = None, None
    if args.rich:
        import spacy
        from scripts.analyze_instruction_landmark_association import PublicNames
        nlp = spacy.load(args.spacy_model, disable=["ner"])
        names = PublicNames(public_catalog)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts, reasons = Counter(), Counter()
    rows = []
    allowed = {"map_name", "instruction", "episode", "instruction_index"}
    with args.instructions.open(encoding="utf-8") as stream:
        for row_id, line in enumerate(stream):
            if not line.strip():
                continue
            row = json.loads(line)
            if set(row) - allowed:
                raise ValueError("Input contains private or unsupported fields")
            map_name, text = row["map_name"], row["instruction"]
            if map_name not in public_catalog:
                raise ValueError("Missing map from public catalog: " + map_name)
            result = parse_unified_instruction(
                text, map_name, public_catalog[map_name],
                nlp=nlp, public_names=names,
            )
            counts[result["intent"]] += 1
            if not result["approved_for_inference"]:
                reasons[result["reason"]] += 1
            rows.append(dict(
                row_id=row_id,
                episode=row.get("episode"),
                instruction_index=row.get("instruction_index"),
                parse=result,
            ))

    with args.output.open("x", encoding="utf-8") as stream:
        for item in rows:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary = dict(
        total=len(rows),
        inferred_actions=sum(row["parse"]["approved_for_inference"] for row in rows),
        by_intent=dict(counts),
        abstention_reasons=dict(reasons),
        rich_parser=args.rich,
        human_verified=False,
        target_GT_used=False,
        note="Automatic proposals are not gold target/relation labels",
    )
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
