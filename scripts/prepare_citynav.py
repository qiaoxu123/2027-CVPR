"""Convert original CityNav JSON assets into independent GraphNav JSONL.

Example:
 python scripts/prepare_citynav.py --objects data/cityrefer/objects.json \
  --processed data/cityrefer/processed_descriptions.json \
  --trajectories data/processed_citynav/citynav_train_seenall.json \
  --bounds data/citynav_bounds.json --output data/graphnav_train_seen.jsonl

map bounds are PUBLIC map metadata (not target-dependent). If ground levels
are unavailable, observation coverage masks are empty, never approximated
from unobserved future RGB.
"""

from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path

import cv2
import numpy as np

from graphnav.data import bind_reference_landmarks, resample_teacher_path
from graphnav.landmark_alias import build_name_index
from graphnav.geometry import normalize_xy
from graphnav.memory import ObservedRGBMemory


def yaw_of_pose(pose):
    if len(pose) == 5:
        return float(pose[3])
    if len(pose) == 6:
        return float(np.arctan2(pose[4], pose[3]))
    raise ValueError("CityNav trajectory pose must have 5 or 6 values")


def masks_at_prefix(path, step, bounds, ground_level, grid):
    """Compute current/explored visibility from observed poses up to t ONLY."""
    if ground_level is None:
        return np.zeros((2, grid, grid), dtype=np.float32)
    memory = ObservedRGBMemory(bounds, size=max(128, grid * 8))
    explored = np.zeros((grid, grid), np.float32)
    current = np.zeros((grid, grid), np.float32)
    for j in range(step + 1):
        pose = path[j]
        altitude = float(pose[2]) - float(ground_level)
        if altitude <= 0:
            continue
        corners = memory.view_corners((pose[0], pose[1]), yaw_of_pose(pose), altitude)
        uv = memory.world_to_image(corners)
        footprint = cv2.fillConvexPoly(
            np.zeros((memory.size, memory.size), dtype=np.uint8),
            np.rint(uv).astype(np.int32), 1,
        )
        coverage = cv2.resize(footprint, (grid, grid), interpolation=cv2.INTER_AREA)
        explored = np.maximum(explored, coverage)
        current = coverage
    return np.stack((current, explored), axis=0)


def convert(objects, processed, trajectories, bounds, output,
            grid=28, stride=4, waypoints=8, max_landmarks=16,
            ground_levels=None, max_steps=20, renderer=None, static_only=False,
            dataset_version="unspecified", match_mode="typed", rgb_observation_step=None):
    count, skipped = 0, 0
    indexed_names = {area: build_name_index(entries) for area, entries in objects.items()}
    with Path(output).open("w", encoding="utf-8") as f:
        for episode in trajectories:
            map_name = f"{episode['area']}_block_{episode['block']}"
            try:
                b = bounds[map_name]
                obj_id = str(episode["object_ids"][0])
                ann_id = int(episode["ann_ids"][0])
                target_obj = objects[map_name][obj_id]
                desc = target_obj["descriptions"][ann_id]
                relation = processed[map_name][obj_id][ann_id]
                target_world = np.asarray(episode["target_positions"][-1][:2], np.float32)
                target_norm = normalize_xy(target_world, b).tolist()
                references = bind_reference_landmarks(
                    relation.get("landmarks", []), objects[map_name], max_landmarks,
                    allowed_types=None, index=indexed_names[map_name], match_mode=match_mode,
                )
                landmarks = [
                    {
                        "id": lm["id"], "name": lm["name"],
                        "object_type": lm["object_type"], "aliases": lm.get("aliases", []),
                        "contour": normalize_xy(lm["contour"], b).tolist(),
                    }
                    for lm in references
                ]
                if static_only:
                    observed_prefix = []
                    if renderer is not None:
                        if rgb_observation_step is None:
                            raise ValueError("static RGB requires explicit --rgb-observation-step")
                        poses = episode.get("trajectory", [])
                        if not poses or rgb_observation_step < 0 or rgb_observation_step >= len(poses):
                            raise ValueError("no valid CityFlight observed prefix at requested step")
                        ground = (ground_levels or {}).get(map_name)
                        if ground is None:
                            raise ValueError("rendered RGB requires map ground level")
                        episode_key = f"{obj_id}_{ann_id}_{count}"
                        for t in range(rgb_observation_step + 1):
                            observed_prefix.append(renderer.render(
                                map_name, episode_key, t, poses[t], ground
                            ))
                    record = {
                        "dataset_version": dataset_version,
                        "match_mode": match_mode,
                        "rgb_observations": observed_prefix,
                        "observation_cutoff_step": rgb_observation_step if renderer is not None else None,
                        "episode_id": f"{map_name}:{obj_id}:{ann_id}:0",
                        "map_name": map_name,
                        "world_bounds": list(map(float, b)),
                        "instruction": desc, "landmarks": landmarks,
                        "target_xy": target_norm,
                        "physical_map_sizes": [float(b[2]-b[0]), float(b[3]-b[1])],
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
                    continue
                human = np.asarray([p[:2] for p in episode["trajectory"]], np.float32)
                human_norm = normalize_xy(human, b)
                poses = episode["trajectory"]
                last = min(len(poses), max_steps)
                if last < 1:
                    raise ValueError("episode missing poses")
                observed_prefix = []
                if renderer is not None:
                    ground = (ground_levels or {}).get(map_name)
                    if ground is None:
                        raise ValueError("RGB rendering needs ground level for this map")
                    episode_key = f"{obj_id}_{ann_id}_{count}"
                    for t in range(last):
                        observed_prefix.append(
                            renderer.render(map_name, episode_key, t, poses[t], ground)
                        )
                for step in range(0, last, stride):
                    pose = poses[step]
                    xy = normalize_xy(pose[:2], b)
                    yaw = yaw_of_pose(pose)
                    future = resample_teacher_path(human_norm, step, waypoints)
                    maps = masks_at_prefix(
                        poses, step, b,
                        (ground_levels or {}).get(map_name), grid,
                    )
                    record = {
                        "dataset_version": dataset_version, "match_mode": match_mode,
                        "observation_cutoff_step": step,
                        "episode_id": f"{map_name}:{obj_id}:{ann_id}:{step}",
                        "map_name": map_name,
                        "world_bounds": list(map(float, b)),
                        # These are ONLY images from poses <= current time.
                        "rgb_observations": observed_prefix[:step + 1],
                        "instruction": desc,
                        "landmarks": landmarks,
                        "uav_pose": [float(xy[0]), float(xy[1]), float(np.sin(yaw)), float(np.cos(yaw))],
                        "spatial_maps": maps.tolist(),
                        "target_xy": target_norm,
                        "teacher_future": future.tolist(),
                        "physical_map_sizes": [float(b[2] - b[0]), float(b[3] - b[1])],
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
            except (KeyError, IndexError, ValueError) as exc:
                skipped += 1
                print(f"Skip malformed episode {map_name}: {exc}")
    return count, skipped


def main():
    p = argparse.ArgumentParser()
    for name in ("objects", "processed", "trajectories", "bounds", "output"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--ground-levels", default=None)
    p.add_argument("--rgbd-dir", default=None,
                   help="CityNav png+tif directory, only used by OFFLINE view renderer")
    p.add_argument("--observations-out", default=None,
                   help="Directory for causal observed-view PNG frames")
    p.add_argument("--grid", type=int, default=28)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--waypoints", type=int, default=8)
    p.add_argument("--max-landmarks", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--dataset-version", choices=("original", "revised"), required=True,
                   help="Declare exact CityNav version of ALL supplied source assets")
    p.add_argument("--match-mode", choices=("exact", "typed", "fuzzy"),
                   default="typed")
    p.add_argument("--rgb-observation-step", type=int, default=None,
                   help="For --static-only + RGB, observed CityFlight prefix 0..t inclusive")
    p.add_argument("--static-only", action="store_true",
                   help="One query per record; static inputs exclude pose/trajectory; RGB requires causal prefix")
    args = p.parse_args()
    def load(path):
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    renderer = None
    if bool(args.rgbd_dir) != bool(args.observations_out):
        p.error("--rgbd-dir and --observations-out must both be supplied")
    if args.rgbd_dir:
        if not args.ground_levels:
            p.error("offline RGB rendering requires --ground-levels")
        from graphnav.renderer import CityNavObservedViewRenderer
        renderer = CityNavObservedViewRenderer(args.rgbd_dir, args.observations_out)
    count, skipped = convert(
        load(args.objects), load(args.processed), load(args.trajectories),
        load(args.bounds), args.output,
        grid=args.grid, stride=args.stride, waypoints=args.waypoints,
        max_landmarks=args.max_landmarks,
        ground_levels=load(args.ground_levels) if args.ground_levels else None,
        max_steps=args.max_steps, renderer=renderer, static_only=args.static_only,
        dataset_version=args.dataset_version, match_mode=args.match_mode,
        rgb_observation_step=args.rgb_observation_step,
    )
    def sha256_file(filename):
        h = hashlib.sha256()
        with open(filename, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    provenance = {
        "dataset_version_declared": args.dataset_version,
        "match_mode": args.match_mode,
        "static_only": args.static_only,
        "rgb_observation_step": args.rgb_observation_step,
        "source_sha256": {
            key: sha256_file(getattr(args, key))
            for key in ("objects", "processed", "trajectories", "bounds")
        },
        "ground_levels_sha256": sha256_file(args.ground_levels)
        if args.ground_levels else None,
        "rgb_generated_from_observed_prefix_only": bool(renderer),
        "note": (
            "The version is declared, not inferred from file contents. "
            "Compare these checksums to the same Revised CityNav files used by SBF."
        ),
        "records_generated": count,
        "episodes_skipped": skipped,
    }
    Path(str(args.output) + ".provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(f"Prepared {count} causal training steps; skipped {skipped} malformed episodes")


if __name__ == "__main__":
    main()
