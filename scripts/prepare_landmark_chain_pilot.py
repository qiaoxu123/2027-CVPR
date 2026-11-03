"""Capture TRAIN camera observations with explicit fixed yaw, without GT-guided poses.

Task categories/instance IDs stratify offline sampling only. Observation steps are
uniform trajectory indices, never chosen by target distance. Public crops and
inference metadata are separated from loss/evaluation labels on disk.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
import time

import cv2
import numpy as np

from graphnav.checkpointing import file_sha256
from graphnav.fixed_yaw_observation import FixedYawObservation
from graphnav.renderer import CityNavObservedViewRenderer
from graphnav.landmark_projection import world_to_frame, frame_to_world


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--rgbd', type=Path, required=True)
    p.add_argument('--objects', type=Path, required=True)
    p.add_argument('--sbf-source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--per-category', type=int, default=6)
    p.add_argument('--steps', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--yaw-rad', type=float, default=0.)
    a = p.parse_args()
    if min(a.per_category, a.steps) < 1:
        raise ValueError('positive task/frame counts required')
    if a.output.exists():
        raise FileExistsError('refusing to overwrite prior observations')
    sys.path.insert(0, str(a.sbf_source))
    from sbfnav.dataset import CityNavDataset
    from sbfnav.mapdata import GROUND_LEVEL
    ds = CityNavDataset(a.data_root, 'train_seen', version='processed_citynav')
    objects = json.loads(a.objects.read_text())
    groups = defaultdict(list)
    for i, raw in enumerate(ds._raw_records):
        name = f"{raw['area']}_block_{raw['block']}"
        oid = int(raw['object_ids'][0])
        category = objects[name][str(oid)]['object_type']
        groups[(category, name, oid)].append(i)
    def key(group):
        return hashlib.sha256(f'{a.seed}:{group}'.encode()).hexdigest()
    selected = []
    for category in ('Car', 'Building', 'Parking', 'Ground'):
        candidates = sorted((g for g in groups if g[0] == category), key=key)
        chosen = []
        maps = set()
        for group in candidates:
            if group[1] not in maps:
                chosen.append(group)
                maps.add(group[1])
            if len(chosen) == a.per_category:
                break
        if len(chosen) < a.per_category:
            chosen += [g for g in candidates if g not in chosen][:a.per_category-len(chosen)]
        selected += [(g, groups[g][0]) for g in chosen]
    # Map-major ordering avoids repeated orthophoto loads; fixed selection is hash-based.
    selected.sort(key=lambda x: (x[0][1], x[1]))
    a.output.mkdir(parents=True)
    rgb_dir = a.output / 'rgb'
    rgb_dir.mkdir()
    renderer = FixedYawObservation(CityNavObservedViewRenderer(a.rgbd, a.output/'unused', image_size=384), a.yaw_rad)
    public = []
    labels = []
    tasks = []
    manifest = dict(complete=False, split='train_seen', annotation_version='processed_citynav',
                    annotation_sha256=file_sha256(ds.annotation_path), objects_sha256=file_sha256(a.objects),
                    preparation_sha256=file_sha256(__file__), fixed_camera_yaw_rad=a.yaw_rad,
                    image_size=[384,384], steps_selection='uniform index; no GT distance selection',
                    seed=a.seed, selected_physical_instances=len(selected), categories={}, frames=0,
                    warning='TRAIN observation capture only. Trajectories are dataset references; this is NOT closed-loop evaluation.')
    (a.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    started = time.perf_counter()
    for group, index in selected:
        record = ds[index]
        category, map_name, oid = group
        # This task metadata is annotation-side, excluded from inference feature records.
        task_key = f'train_{index:05d}'
        tasks.append(dict(task_key=task_key, episode_index=index, map_name=map_name, category=category,
                          physical_group=[map_name, oid], instruction=record.instruction))
        steps = np.unique(np.linspace(0, len(record.trajectory)-1, a.steps, dtype=int))
        for step in steps:
            pose = record.trajectory[step]
            if pose[2] <= GROUND_LEVEL[map_name]:
                labels.append(dict(task_key=task_key, step=int(step), status='invalid_altitude'))
                continue
            obs = renderer.current_rgbd(map_name, pose, GROUND_LEVEL[map_name])
            frame_key = f'{task_key}_{step:04d}'
            image = rgb_dir / f'{frame_key}.png'
            if not cv2.imwrite(str(image), obs['bgr']):
                raise IOError(image)
            np.savez_compressed(rgb_dir/f'{frame_key}_depth.npz', depth_m=obs['depth_m'], depth_valid=obs['depth_valid'])
            pixels = np.array([[0.,0.], [383.,383.], [123.,251.]], np.float32)
            recovered = world_to_frame(frame_to_world(pixels, obs['footprint_world'], obs['bgr'].shape), obs['footprint_world'], obs['bgr'].shape)
            public.append(dict(frame_key=frame_key, task_key=task_key, map_name=map_name,
                               instruction=record.instruction, step=int(step), agent_pose=pose.tolist(),
                               camera_pose=obs['pose'].tolist(), footprint_world=obs['footprint_world'].tolist(),
                               image=str(image.relative_to(a.output)), image_sha256=file_sha256(image),
                               depth=str((rgb_dir/f'{frame_key}_depth.npz').relative_to(a.output)),
                               depth_sha256=file_sha256(rgb_dir/f'{frame_key}_depth.npz'),
                               valid_fraction=float(obs['depth_valid'].mean()),
                               pixel_world_roundtrip_max_px=float(np.abs(recovered-pixels).max())))
            labels.append(dict(frame_key=frame_key, task_key=task_key, status='captured',
                               target_world=record.target_position.tolist(), category=category,
                               physical_group=[map_name,oid]))
        print(json.dumps(dict(task=task_key, map=map_name, frames=len(public), seconds=time.perf_counter()-started)), flush=True)
    for name, values in [('public_frames.jsonl', public), ('labels_private.jsonl', labels), ('tasks_private.json', tasks)]:
        path = a.output/name
        path.write_text(json.dumps(values, ensure_ascii=False, indent=2)+'\n' if name.endswith('.json') else ''.join(json.dumps(v, ensure_ascii=False)+'\n' for v in values))
        manifest[name+'_sha256'] = file_sha256(path)
    manifest.update(complete=True, frames=len(public), seconds=time.perf_counter()-started,
                    maps=len({g[1] for g,_ in selected}), categories=dict(Counter(g[0] for g,_ in selected)),
                    max_roundtrip_error_px=max(x['pixel_world_roundtrip_max_px'] for x in public))
    (a.output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
