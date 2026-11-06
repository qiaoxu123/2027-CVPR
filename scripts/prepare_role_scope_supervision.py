"""Prepare GT-free public role weak labels for the existing 2026-11-06 relation study.

Reads ONLY parsed language, public-input scenes, and fit/dev split indices.
No target XY, GT instance ID, angle bins, private confirmation labels or RGB.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from graphnav.checkpointing import atomic_json, atomic_save, file_sha256
from graphnav.relation_scope_supervision import public_role_targets


def prepare(run: Path, output: Path):
    public_path = run / "public_inputs.pt"
    parser_path = run / "data/parsed.jsonl"
    split_path = run / "data/split.json"
    for path in (public_path, parser_path, split_path):
        if not path.exists():
            raise FileNotFoundError(path)
    data = torch.load(public_path, map_location="cpu", weights_only=True)
    parsed = [json.loads(line) for line in parser_path.read_text().splitlines() if line.strip()]
    split = json.loads(split_path.read_text())
    if len(parsed) != len(data["queries"]):
        raise ValueError("Parser and public scene query counts differ")
    fit = list(split["indices"]["fit"])
    dev = list(split["indices"]["dev"])
    if set(fit) & set(dev):
        raise ValueError("TRAIN fit and dev overlap")
    weak = {}
    by_split = {}
    for name, indices in (("fit", fit), ("dev", dev)):
        counter = Counter()
        labeled_queries = 0
        for i in indices:
            row = parsed[i]
            query = data["queries"][i]
            scene = data["scenes"][query["scene"]]
            if row["index"] != i or row["map_name"] != scene["map_name"]:
                raise ValueError("Parser/public-scene alignment mismatch")
            targets, reasons = public_role_targets(row, scene, query)
            weak[i] = targets
            counter.update(reasons)
            if targets:
                labeled_queries += 1
        by_split[name] = dict(queries=len(indices),
                              queries_with_public_weak_roles=labeled_queries,
                              coverage=labeled_queries / len(indices),
                              labels_by_kind=dict(counter))
    output.mkdir(parents=True, exist_ok=False)
    atomic_save(dict(format="public-role-weak-targets-v1", fit=fit, dev=dev,
                     targets=weak), output / "PUBLIC_weak_role_targets.pt")
    report = dict(format="public-role-weak-targets-v1", splits=by_split,
                  public_inputs_sha256=file_sha256(public_path),
                  parser_sha256=file_sha256(parser_path),
                  split_sha256=file_sha256(split_path),
                  generator_sha256=file_sha256(__file__),
                  no_GT_labels_read=True, no_confirm_split_read=True,
                  public_roles_are_weak_not_instance_ground_truth=True)
    atomic_json(report, output / "PUBLIC_role_coverage.json")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True,
                   help="Existing relation_bearing_s42_20261106 GPU run directory")
    p.add_argument("--output", type=Path, required=True,
                   help="New, nonexisting directory outside historical artifacts")
    args = p.parse_args()
    print(json.dumps(prepare(args.run, args.output), indent=2), flush=True)


if __name__ == "__main__":
    main()
