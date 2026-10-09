"""Paired metric audits and training tables for the completed A0-A5 run."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def paired(a, b):
    assert [(r["episode_id"], r["instruction"]) for r in a] == [(r["episode_id"], r["instruction"]) for r in b]
    rng = np.random.default_rng(0)
    difference = np.asarray([float(y["hits"]["8"]) - float(x["hits"]["8"]) for x, y in zip(a, b)])
    maps = np.asarray([r["episode_id"].split(":")[0] for r in a])
    names = sorted(set(maps))
    sums = np.asarray([difference[maps == name].sum() for name in names])
    counts = np.asarray([(maps == name).sum() for name in names])
    independent, clustered = [], []
    for _ in range(2000):
        independent.append(difference[rng.integers(0, len(a), len(a))].mean() * 100)
        chosen = rng.integers(0, len(names), len(names))
        clustered.append(sums[chosen].sum() / counts[chosen].sum() * 100)
    return {"R8_difference_pp": float(difference.mean() * 100),
            "paired_query_bootstrap_95ci_pp": np.quantile(independent, [.025, .975]).tolist(),
            "map_cluster_bootstrap_95ci_pp": np.quantile(clustered, [.025, .975]).tolist(),
            "maps": len(names), "queries": len(a),
            "R8_gained_queries": int((difference > 0).sum()),
            "R8_lost_queries": int((difference < 0).sum())}


def summarize(run, output):
    run, output = Path(run), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = json.loads((run / "full_localization_metrics.json").read_text())
    training = []
    for variant, logs in report["training"].items():
        for log in logs:
            row = {k: log[k] for k in ("variant", "epoch", "loss", "train_seconds",
                "train_samples", "samples_per_second", "peak_gpu_gib", "valid_visual_candidates_train")}
            row.update({"val_seen_" + k: log["val_seen"][k] for k in ("R@1/20m", "R@4/20m", "R@8/20m", "NE_m")})
            row["train_visual_candidate_fraction"] = log["valid_visual_candidates_train"] / max(1, log["train_samples"] * report["protocol"]["top_k"])
            training.append(row)
    with (output / "training_epochs.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(training[0]))
        writer.writeheader()
        writer.writerows(training)
    diagnostics = {"paired_comparisons": {}, "rgb_effect": {}, "failure_groups": {}}
    for split in ("val_seen", "val_unseen"):
        data = {v: rows(run / f"{v}_{split}_queries.jsonl") for v in report["metrics"]}
        for variant, records in data.items():
            metric = report["metrics"][variant][split]
            assert len(records) == metric["samples"]
            for k in (1, 4, 8):
                assert abs(np.mean([r["hits"][str(k)] for r in records]) - metric[f"R@{k}/20m"]) < 1e-12
            assert abs(np.mean([r["NE_m"] for r in records]) - metric["NE_m"]) < 1e-4
        diagnostics["paired_comparisons"][split] = {
            f"{a}_to_{b}": paired(data[a], data[b]) for a, b in (
                ("rules", "tokens"), ("rules", "geometry"), ("rules", "gat"),
                ("geometry", "gat"), ("gat", "rgb"))}
        no_rgb = [i for i, row in enumerate(data["rgb"]) if not any(v >= .3 for v in row["visual_coverage"])]
        mismatches = sum(any(data["gat"][i]["hits"][str(k)] != data["rgb"][i]["hits"][str(k)] for k in (1, 4, 8))
            or abs(data["gat"][i]["NE_m"] - data["rgb"][i]["NE_m"]) > 1e-5 for i in no_rgb)
        diagnostics["rgb_effect"][split] = {"queries_with_no_valid_rgb": len(no_rgb),
            "no_rgb_predictions_changed": mismatches,
            "A4_R20": np.mean([r["hits"]["20"] for r in data["gat"]]),
            "A5_R20": np.mean([r["hits"]["20"] for r in data["rgb"]])}
        groups = {}
        for variant in ("rules", "tokens", "geometry", "gat", "rgb"):
            groups[variant] = {}
            for key, condition in (("relation_parsed", lambda r: r["relation_count"] > 0),
                                   ("no_relation", lambda r: r["relation_count"] == 0)):
                selected = [r for r in data[variant] if condition(r)]
                groups[variant][key] = {"queries": len(selected),
                    "R8": float(np.mean([r["hits"]["8"] for r in selected])),
                    "NE_m": float(np.mean([r["NE_m"] for r in selected]))}
        diagnostics["failure_groups"][split] = groups
    (output / "paired_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    for filename in ("full_localization_metrics.json", "full_localization_metrics.csv"):
        (output / filename).write_bytes((run / filename).read_bytes())
    return diagnostics


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    print(json.dumps(summarize(args.run, args.output)["rgb_effect"]))
