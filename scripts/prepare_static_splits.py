"""Deduplicate observable queries and remove train/validation text overlap."""

import argparse
import json
from pathlib import Path

import numpy as np


def prepare(data_dir, output):
    root = Path(data_dir)
    data = {}
    report = {}
    for split in ("train_seen", "val_seen", "val_unseen"):
        path = root / f"graphnav_{split}.jsonl"
        raw = path.with_suffix(".raw.jsonl")
        if not raw.exists():
            raw.write_bytes(path.read_bytes())
        records = [json.loads(l) for l in raw.read_text().splitlines()]
        unique = {}
        duplicate_ids = []
        target_distances = []
        for record in records:
            key = (record["map_name"], record["instruction"])
            if key in unique:
                duplicate_ids.append(record["episode_id"])
                a = unique[key]
                target_distances.append(
                    float(
                        np.linalg.norm(
                            (np.asarray(a["target_xy"]) - np.asarray(record["target_xy"]))
                            * np.asarray(record["physical_map_sizes"])
                        )
                    )
                )
            else:
                unique[key] = record
        data[split] = unique
        report[split] = {
            "raw_count": len(records),
            "deduplicated_count": len(unique),
            "duplicate_episode_ids": duplicate_ids,
            "duplicates_with_different_label": sum(d > 0.001 for d in target_distances),
            "max_duplicate_target_distance_m": max(target_distances, default=0.0),
        }
    heldout = set(data["val_seen"]) | set(data["val_unseen"])
    removed = [
        data["train_seen"][key]["episode_id"] for key in data["train_seen"] if key in heldout
    ]
    data["train_seen"] = {key: r for key, r in data["train_seen"].items() if key not in heldout}
    report["train_validation_overlap_removed"] = removed
    report["policy"] = (
        "Exact map+instruction; first source record retained, never target-based. Validation preserved; train overlaps removed by text only."
    )
    for split, records in data.items():
        (root / f"graphnav_{split}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records.values())
        )
        report[split]["final_count"] = len(records)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print({split: len(r) for split, r in data.items()}, "train overlaps removed", len(removed))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    prepare(a.data_dir, a.output)
