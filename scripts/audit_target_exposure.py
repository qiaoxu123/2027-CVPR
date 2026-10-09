"""Post-hoc Seen/Train target FOV diagnostics; never used as model inputs."""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from graphnav.landmark_projection import world_to_frame


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('observations', 'sbf-source', 'data-root', 'output'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--split', choices=('train_seen', 'val_seen'), required=True)
    args = p.parse_args(); output = Path(args.output)
    if output.exists():
        raise ValueError('refusing diagnostic overwrite')
    sys.path.insert(0, args.sbf_source)
    from sbfnav.dataset import CityNavDataset
    dataset = CityNavDataset(args.data_root, args.split, version='processed_citynav')
    frames = {}; rows = []; episode = {}
    for line in Path(args.observations).open():
        record = json.loads(line); index = record['source_episode_index']
        source = dataset[index]
        if source.map_name != record['map_name']:
            raise ValueError('source episode index/map mismatch')
        for observation in record['rgb_observations']:
            key = (index, observation['step'], observation['path'])
            if key in frames:
                continue
            if observation['step'] > record['observation_cutoff_step']:
                raise ValueError('future RGB in causal prefix')
            image = cv2.imread(observation['path'])
            if image is None:
                raise ValueError('missing actual RGB')
            height, width = image.shape[:2]
            xy = world_to_frame(source.target_position[None, :2], observation['footprint_world'], image.shape)[0]
            in_fov = bool(0 <= xy[0] <= width-1 and 0 <= xy[1] <= height-1)
            supported = in_fov and bool(image[int(round(xy[1])), int(round(xy[0]))].max() > 0)
            pose = source.trajectory[observation['step']]
            distance = float(np.linalg.norm(pose[:2] - source.target_position[:2]))
            row = dict(source_episode_index=index, map_name=source.map_name, step=observation['step'],
                target_in_geometric_fov=int(in_fov), target_point_rgb_nonzero=int(supported),
                target_xy_distance_m=distance, native_width=width, native_height=height,
                distance_bin='<=50m' if distance <= 50 else ('50-100m' if distance <= 100 else '>100m'))
            rows.append(row); frames[key] = True
            state = episode.setdefault(index, dict(any_fov=False, any_support=False, first_fov=False, first_support=False))
            state['any_fov'] |= in_fov; state['any_support'] |= supported
            if observation['step'] == 0:
                state['first_fov'] = in_fov; state['first_support'] = supported
    def summarize(items):
        return dict(frames=len(items), geometric_fov_fraction=sum(r['target_in_geometric_fov'] for r in items)/len(items),
            nonzero_target_pixel_fraction=sum(r['target_point_rgb_nonzero'] for r in items)/len(items))
    summary = dict(split=args.split, source_sha256=hashlib.sha256(dataset.annotation_path.read_bytes()).hexdigest(),
        observations_sha256=hashlib.sha256(Path(args.observations).read_bytes()).hexdigest(),
        coverage=summarize(rows), source_episodes_total=len(dataset), observed_episodes=len(episode),
        episode_fractions={k: sum(e[k] for e in episode.values())/len(episode) for k in next(iter(episode.values()))},
        by_distance={k: summarize([r for r in rows if r['distance_bin']==k]) for k in sorted({r['distance_bin'] for r in rows})},
        resolutions=dict(Counter(f"{r['native_width']}x{r['native_height']}" for r in rows)),
        warning='OFFLINE LABEL-BASED DIAGNOSTIC ONLY. Geometric FOV plus nonblack source pixel is not semantic visibility or occlusion ground truth. Human trajectory frames are subsampled/capped by existing P0 manifest and are not GraphNav on-policy exposure. No GT field enters inference or visual training.')
    output.mkdir(parents=True)
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    with (output/'frames.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
