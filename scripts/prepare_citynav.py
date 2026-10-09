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
from pathlib import Path
import cv2
import numpy as np

from graphnav.data import bind_reference_landmarks, resample_teacher_path
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
            ground_levels=None, max_steps=20):
    count, skipped = 0, 0
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
                human = np.asarray([p[:2] for p in episode["trajectory"]], np.float32)
                human_norm = normalize_xy(human, b)
                references = bind_reference_landmarks(
                    relation.get("landmarks", []), objects[map_name], max_landmarks
                )
                landmarks = [
                    {
                        "id": lm["id"], "name": lm["name"],
                        "contour": normalize_xy(lm["contour"], b).tolist(),
                    }
                    for lm in references
                ]
                poses = episode["trajectory"]
                last = min(len(poses), max_steps)
                if last < 1:
                    raise ValueError("episode missing poses")
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
                        "episode_id": f"{map_name}:{obj_id}:{ann_id}:{step}",
                        "map_name": map_name,
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
    p.add_argument("--grid", type=int, default=28)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--waypoints", type=int, default=8)
    p.add_argument("--max-landmarks", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=20)
    args = p.parse_args()
    def load(path):
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    count, skipped = convert(
        load(args.objects), load(args.processed), load(args.trajectories),
        load(args.bounds), args.output,
        grid=args.grid, stride=args.stride, waypoints=args.waypoints,
        max_landmarks=args.max_landmarks,
        ground_levels=load(args.ground_levels) if args.ground_levels else None,
        max_steps=args.max_steps,
    )
    print(f"Prepared {count} causal training steps; skipped {skipped} malformed episodes")


if __name__ == "__main__":
    main()
