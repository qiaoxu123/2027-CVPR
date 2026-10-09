"""Reproducible map-balanced causal camera replay, with no goal fields exported."""
import argparse
from collections import defaultdict, Counter
import hashlib
import json
from pathlib import Path
import random
import time

from graphnav.renderer import CityNavObservedViewRenderer
from graphnav.bidirectional import split_map_names


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("trajectories", "source-split", "ground", "rgbd", "output", "frames"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--episodes-per-map", type=int, default=128)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    allowed = split_map_names(args.source_split)
    grouped = defaultdict(list)
    for i, episode in enumerate(json.loads(Path(args.trajectories).read_text())):
        name = f"{episode['area']}_block_{episode['block']}"
        if name in allowed:
            grouped[name].append((i, episode))
    ground = json.loads(Path(args.ground).read_text())
    renderer = CityNavObservedViewRenderer(args.rgbd, args.frames, image_size=384)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    counts, start = Counter(), time.perf_counter()
    selected = {}
    with out.open("w") as handle:
        for name, episodes in sorted(grouped.items()):
            rng = random.Random(f"{args.seed}:{name}")
            rng.shuffle(episodes)
            episodes = episodes[:args.episodes_per_map]
            selected[name] = [i for i, _ in episodes]
            for i, episode in episodes:
                counts["episodes"] += 1
                # Replay poses only; no target, object ID, language or future label.
                for step in range(0, min(args.max_steps, len(episode["trajectory"])), args.stride):
                    pose = episode["trajectory"][step]
                    if pose[2] <= ground[name]:
                        counts["invalid_altitude_frames"] += 1
                        continue
                    obs = renderer.render(name, f"source_{i}", step, pose, ground[name])
                    row = dict(dataset_version="revised", map_name=name,
                               source_episode_index=i, observation_cutoff_step=step,
                               rgb_observations=[obs])
                    handle.write(json.dumps(row) + "\n")
                    counts["frames"] += 1
            print(json.dumps({"map": name, **counts}), flush=True)
    summary = {**counts, "seconds": time.perf_counter()-start,
               "native_resolution": 384, "args": vars(args), "selected_source_indices": selected,
               "source_sha256": {k: hashlib.sha256(Path(getattr(args, k)).read_bytes()).hexdigest()
                                 for k in ("trajectories", "source_split", "ground")},
               "sampling": "seeded uniform episode sample per map, independent of target/landmark/visibility"}
    Path(str(out) + ".manifest.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
