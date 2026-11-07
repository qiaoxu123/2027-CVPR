"""Paired full Val-Seen rollout for V2 Geometry with public-name bindings.

Only the public-text-to-landmark binding differs between arms. V2 Geometry,
the original frozen Stage1 Trajectory/Stop head, start poses, action protocol,
and scoring are identical. Targets are accessed only after each rollout.
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from graphnav.checkpointing import atomic_json, file_sha256
from graphnav.closedloop import (
    ClosedLoopProtocol, PublicEpisode, PublicCatalog, aggregate, rollout, score_trace,
)
from scripts.evaluate_v2_public_navigation import FrozenV2GeometryPolicy


def load_jsonl(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len(rows) != 2470:
        raise ValueError(f"Expected 2470 public query rows at {path}, found {len(rows)}")
    if any(row.get("target_xy") != [0.0, 0.0] for row in rows):
        raise ValueError("Query file must contain only the neutral target placeholder")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "data", "sbf-source", "config", "source-v2", "trained-v2", "stage1",
        "legacy-queries", "expanded-queries", "output",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--candidate-head-checkpoint", type=Path,
                   help="Optional TRAIN-adapted frozen candidate head; requires --candidate-selection ranked-once")
    p.add_argument("--smoke", type=int, default=0, help="Run the first N episodes only")
    p.add_argument("--candidate-selection", choices=("geometry-top1", "ranked-once", "rank-score-once"), default="geometry-top1",
                   help="Use Geometry slot 0, joint rank/mode selection once, or candidate-score-only selection once")
    p.add_argument("--arms", nargs="+", choices=("legacy", "expanded"), default=("legacy", "expanded"))
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()

    sys.path.insert(0, str(a.sbf_source.resolve()))
    from sbfnav.config import load_config, apply_overrides, validate_config
    from sbfnav.dataset import CityNavDataset, CityReferCatalog
    from sbfnav.planning_state import PlanningStateBuilder
    from sbfnav.navigation import BoundedWaypointController
    from sbfnav.mapdata import GROUND_LEVEL
    from graphnav.dynamic_runtime import NativeObservationBuilder

    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if a.candidate_head_checkpoint and a.candidate_selection not in ("ranked-once", "rank-score-once"):
        raise ValueError("--candidate-head-checkpoint requires a ranked candidate-selection mode")

    config = apply_overrides(load_config(a.config), ["data.root=" + str(a.data)])
    validate_config(config)
    nav, alt = config["navigation"], config["altitude"]
    protocol = ClosedLoopProtocol(
        horizon=nav["horizon"], horizontal_step_m=nav["horizontal_step_m"],
        vertical_step_m=nav["vertical_step_m"],
        min_altitude_m=alt["min_above_ground_m"],
        max_altitude_m=alt["max_above_ground_m"],
        stagnation_steps=nav["stagnation_steps"],
        success_radius_m=nav["success_distance_m"],
    )
    if protocol != ClosedLoopProtocol():
        raise ValueError(f"Expected original shared protocol, got {protocol}")

    ds = CityNavDataset(a.data, "val_seen", version="processed_citynav")
    if len(ds) != 2470:
        raise ValueError(f"Full Val-Seen required, found {len(ds)}")
    legacy_queries = load_jsonl(a.legacy_queries)
    expanded_queries = load_jsonl(a.expanded_queries)
    if any(x["instruction"] != y["instruction"] or x["map_name"] != y["map_name"]
           for x, y in zip(legacy_queries, expanded_queries)):
        raise ValueError("Paired query rows differ in map or original instruction")

    catalog_dir = a.data / "cityrefer"
    catalog_path = catalog_dir / "objects.json"
    catalog = CityReferCatalog(catalog_dir)
    output = a.output
    output.mkdir(parents=True, exist_ok=True)
    expected = 2470 if not a.smoke else min(a.smoke, 2470)
    manifest = {
        "split": "val_seen", "expected_per_arm": expected, "full_split_size": len(ds),
        "protocol": protocol.__dict__, "seed": 42, "precision": "FP32; TF32 disabled",
        "geometry_checkpoint_sha256": file_sha256(a.trained_v2),
        "geometry_source_sha256": file_sha256(a.source_v2),
        "controller_checkpoint_sha256": file_sha256(a.stage1),
        "legacy_queries_sha256": file_sha256(a.legacy_queries),
        "expanded_queries_sha256": file_sha256(a.expanded_queries),
        "annotations_sha256": file_sha256(ds.annotation_path),
        "catalog_sha256": file_sha256(catalog_path),
        "bindings": {
            "legacy": "Existing exact public-name matcher/query cache",
            "expanded": "Frozen PublicLandmarkMentions aliases and add_mention_aliases query cache",
        },
        "candidate_selection": a.candidate_selection,
        "candidate_head_checkpoint_sha256": file_sha256(a.candidate_head_checkpoint) if a.candidate_head_checkpoint else None,
        "controller": f"Original GraphNav Stage1 trajectory/stop weights; threshold .3; {a.candidate_selection}; guarded-blend; no exploration",
        "visual_fusion": False, "OSM": False, "GT_access": "Scoring only after rollout",
        "runner_sha256": file_sha256(Path(__file__)),
        "source_sha256": {
            "closedloop": file_sha256(Path("graphnav/closedloop.py")),
            "a3_runtime": file_sha256(Path("graphnav/a3_runtime.py")),
            "candidate_geometry": file_sha256(Path("graphnav/candidate_geometry.py")),
            "renderer": file_sha256(Path("graphnav/renderer.py")),
        },
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if not a.resume or json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Refuse overwrite/resume with a changed manifest")
    else:
        atomic_json(manifest, manifest_path)

    completed_by_arm = {}
    query_by_arm = {"legacy": legacy_queries, "expanded": expanded_queries}
    for arm in a.arms:
        queries = query_by_arm[arm]
        arm_out = output / arm
        rows_dir = arm_out / "rows"
        rows_dir.mkdir(parents=True, exist_ok=True)
        existing = {int(path.stem) for path in rows_dir.glob("*.json")}
        if any(i >= expected for i in existing):
            raise ValueError(f"Unexpected episode index in {arm_out}")
        if not a.resume and existing:
            raise ValueError(f"Existing rows require --resume: {arm_out}")

        builder_base = PlanningStateBuilder(
            a.data, PublicCatalog(catalog), canvas_size=config["model"]["canvas_size"],
            field_size=config["model"]["field_size"],
            max_height_above_ground_m=config["state"]["max_height_above_ground_m"],
            trajectory_radius_m=config["state"]["trajectory_radius_m"],
        )
        builder = NativeObservationBuilder(builder_base, a.data / "rgbd", output / "observations")
        controller = BoundedWaypointController(
            horizontal_step_m=protocol.horizontal_step_m,
            vertical_step_m=protocol.vertical_step_m,
            minimum_above_ground_m=protocol.min_altitude_m,
            maximum_above_ground_m=protocol.max_altitude_m,
        )
        runtime = FrozenV2GeometryPolicy(
            a.source_v2, a.trained_v2, a.stage1, arm_out / "public_inputs", "trained_v2",
        )
        if a.candidate_selection == "ranked-once":
            runtime.execution = "ranked-guarded-blend"
        elif a.candidate_selection == "rank-score-once":
            runtime.execution = "rank-score-guarded-blend"
        if a.candidate_head_checkpoint:
            candidate_head = torch.load(a.candidate_head_checkpoint, map_location="cpu", weights_only=True)
            if candidate_head.get("geometry_sha256") != file_sha256(a.trained_v2):
                raise ValueError("Candidate head was trained for a different V2 Geometry checkpoint")
            if int(candidate_head.get("hidden", -1)) != int(runtime.args.hidden):
                raise ValueError("Candidate head hidden size differs from current V2")
            runtime.head.load_state_dict(candidate_head["model"], strict=True)
        if (runtime.stop_threshold, runtime.execution, runtime.explore_candidates) != (
            .3, ("ranked-guarded-blend" if a.candidate_selection == "ranked-once" else
                 "rank-score-guarded-blend" if a.candidate_selection == "rank-score-once" else "guarded-blend"), 0,
        ):
            raise ValueError("Controller execution differs from original frozen protocol")
        new_count = 0
        started = time.perf_counter()
        for i in range(expected):
            row_path = rows_dir / f"{i:05d}.json"
            if row_path.exists():
                continue
            query = queries[i]
            refs = tuple(dict.fromkeys(x["name"] for x in query["landmarks"]))
            episode = PublicEpisode(
                query["map_name"], query["instruction"], ds[i].start_pose.copy(), (), refs,
            )
            # Exact frozen public query used by the localization evaluation;
            # private target coordinates/IDs do not enter inference.
            runtime.record = query
            runtime.visited_goal_cells = set()
            runtime.selected_goal_slot = None
            poses, termination, timings, diagnostics = rollout(
                episode, builder, controller, runtime.predict,
                GROUND_LEVEL[query["map_name"]], protocol,
                synchronize=torch.cuda.synchronize,
            )
            # The private endpoint and expert trajectory are first read for scoring.
            metrics = score_trace(
                poses, ds[i].target_position, ds[i].trajectory, termination,
                radius=protocol.success_radius_m, decision_seconds=timings,
            )
            atomic_json({
                "index": i, "episode_id": list(ds[i].episode_id), "map_name": query["map_name"],
                "instruction": query["instruction"], "public_reference_names": refs,
                "poses": poses.tolist(), "termination": termination, "metrics": metrics,
                "decision_seconds": timings, "step_diagnostics": diagnostics,
            }, row_path)
            new_count += 1
            if new_count <= 2 or new_count % 20 == 0:
                print(json.dumps({"arm": arm, "completed": i + 1, "expected": expected,
                                  "SR": metrics["SR"], "OSR": metrics["OSR"],
                                  "NE": metrics["NE"], "termination": termination,
                                  "elapsed_s": time.perf_counter() - started}), flush=True)
            if new_count % 50 == 0:
                partial = [json.loads(x.read_text()) for x in sorted(rows_dir.glob("*.json"))]
                atomic_json({"completed": len(partial), "expected": expected,
                             "metrics": aggregate([x["metrics"] for x in partial])},
                            arm_out / "progress.json")

        rows = [json.loads(x.read_text()) for x in sorted(rows_dir.glob("*.json"))]
        if len(rows) != expected:
            raise ValueError(f"Incomplete {arm}: {len(rows)}/{expected}")
        metrics = aggregate([x["metrics"] for x in rows])
        result = {
            "complete": len(rows) == expected, "coverage": {"completed": len(rows), "expected": expected},
            "metrics": metrics,
            "termination_counts": dict(Counter(x["termination"] for x in rows)),
            "elapsed_session_s": time.perf_counter() - started,
            "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
        atomic_json(result, arm_out / ("metrics.json" if expected == 2470 else "smoke_metrics.json"))
        completed_by_arm[arm] = rows
        del runtime, builder, controller
        torch.cuda.empty_cache()

    if "legacy" in completed_by_arm and "expanded" in completed_by_arm:
        a_rows, b_rows = completed_by_arm["legacy"], completed_by_arm["expanded"]
        a_success = [x["metrics"]["SR"] for x in a_rows]
        b_success = [x["metrics"]["SR"] for x in b_rows]
        result = {
            "complete": True, "coverage_per_arm": {"legacy": len(a_rows), "expanded": len(b_rows)},
            "legacy": aggregate([x["metrics"] for x in a_rows]),
            "expanded": aggregate([x["metrics"] for x in b_rows]),
            "paired_success": {
                "corrected": sum(y > x for x, y in zip(a_success, b_success)),
                "damaged": sum(y < x for x, y in zip(a_success, b_success)),
                "unchanged": sum(y == x for x, y in zip(a_success, b_success)),
                "net_successes": sum(b_success) - sum(a_success),
            },
            "termination_counts": {
                arm: dict(Counter(x["termination"] for x in rows))
                for arm, rows in (("legacy", a_rows), ("expanded", b_rows))
            },
            "manifest": manifest,
        }
        atomic_json(result, output / "metrics.json")
        print(json.dumps(result, indent=2), flush=True)
    else:
        result = {
            "complete": all(len(rows) == expected for rows in completed_by_arm.values()),
            "coverage_per_arm": {k: len(v) for k, v in completed_by_arm.items()},
            "expected": expected,
            "candidate_selection": a.candidate_selection,
            "arms": {k: aggregate([x["metrics"] for x in v]) for k, v in completed_by_arm.items()},
        }
        if expected == 2470:
            result["manifest"] = manifest
            atomic_json(result, output / "metrics.json")
        else:
            atomic_json(result, output / "smoke_metrics.json")
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
