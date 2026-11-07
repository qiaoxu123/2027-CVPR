"""Create compact paired summaries from the frozen V2 full-navigation run."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from graphnav.closedloop import aggregate


def read_rows(root, arm):
    files = sorted((root / arm / "rows").glob("*.json"))
    rows = [json.loads(path.read_text()) for path in files]
    if len(rows) != 2470 or [row["index"] for row in rows] != list(range(2470)):
        raise ValueError(f"Expected complete ordered 2470 rows for {arm}, found {len(rows)}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    legacy, expanded = read_rows(args.run, "legacy"), read_rows(args.run, "expanded")
    if any(a["episode_id"] != b["episode_id"] or a["index"] != b["index"]
           for a, b in zip(legacy, expanded)):
        raise ValueError("Episode pairing mismatch")

    args.output.mkdir(parents=True, exist_ok=True)
    paired_path = args.output / "paired_episode_outcomes.csv"
    fields = ["index", "episode_id", "map_name", "legacy_SR", "expanded_SR",
              "legacy_OSR", "expanded_OSR", "legacy_NE_m", "expanded_NE_m",
              "legacy_SPL", "expanded_SPL", "legacy_nDTW", "expanded_nDTW",
              "legacy_steps", "expanded_steps", "legacy_termination", "expanded_termination",
              "same_pose_trace"]
    with paired_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for a, b in zip(legacy, expanded):
            x, y = a["metrics"], b["metrics"]
            writer.writerow({
                "index": a["index"], "episode_id": json.dumps(a["episode_id"]),
                "map_name": a["map_name"], "legacy_SR": x["SR"], "expanded_SR": y["SR"],
                "legacy_OSR": x["OSR"], "expanded_OSR": y["OSR"],
                "legacy_NE_m": x["NE"], "expanded_NE_m": y["NE"],
                "legacy_SPL": x["SPL"], "expanded_SPL": y["SPL"],
                "legacy_nDTW": x["nDTW"], "expanded_nDTW": y["nDTW"],
                "legacy_steps": x["steps"], "expanded_steps": y["steps"],
                "legacy_termination": a["termination"],
                "expanded_termination": b["termination"],
                "same_pose_trace": int(a["poses"] == b["poses"]),
            })

    def grouped(rows):
        out = defaultdict(list)
        for row in rows:
            out[row["map_name"]].append(row["metrics"])
        return {name: {k: v for k, v in aggregate(values).items() if k != "metric_units"}
                for name, values in sorted(out.items())}

    by_map_legacy, by_map_expanded = grouped(legacy), grouped(expanded)
    map_path = args.output / "navigation_by_map.csv"
    with map_path.open("w", newline="") as stream:
        map_fields = ["map_name", "episodes"]
        for metric in ("SR", "OSR", "SPL", "NE", "nDTW", "steps", "inference_ms"):
            map_fields.extend((f"legacy_{metric}", f"expanded_{metric}", f"delta_{metric}"))
        writer = csv.DictWriter(stream, fieldnames=map_fields)
        writer.writeheader()
        for name in sorted(by_map_legacy):
            row = {"map_name": name, "episodes": by_map_legacy[name]["episodes"]}
            for metric in ("SR", "OSR", "SPL", "NE", "nDTW", "steps", "inference_ms"):
                a, b = by_map_legacy[name][metric], by_map_expanded[name][metric]
                row.update({f"legacy_{metric}": a, f"expanded_{metric}": b,
                            f"delta_{metric}": b - a})
            writer.writerow(row)

    legacy_sr = [row["metrics"]["SR"] for row in legacy]
    expanded_sr = [row["metrics"]["SR"] for row in expanded]
    run_metrics = json.loads((args.run / "metrics.json").read_text())
    output = {
        "complete": True,
        "coverage_per_arm": {"legacy": len(legacy), "expanded": len(expanded)},
        "legacy": run_metrics["legacy"],
        "expanded": run_metrics["expanded"],
        "paired_success": {
            "corrected": sum(b > a for a, b in zip(legacy_sr, expanded_sr)),
            "damaged": sum(b < a for a, b in zip(legacy_sr, expanded_sr)),
            "unchanged": sum(b == a for a, b in zip(legacy_sr, expanded_sr)),
            "net_successes": sum(expanded_sr) - sum(legacy_sr),
            "same_full_pose_trace": sum(a["poses"] == b["poses"] for a, b in zip(legacy, expanded)),
        },
        "by_map": {"legacy": by_map_legacy, "expanded": by_map_expanded},
        "run_manifest": run_metrics["manifest"],
    }
    (args.output / "navigation_summary.json").write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
