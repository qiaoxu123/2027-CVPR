"""TRAIN reference-pose supervision audit, never a navigation inference path."""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from graphnav.checkpointing import atomic_json, file_sha256
from graphnav.observed_candidate_rois import CurrentCandidateROIEnvironment


def reference_indices(trajectory):
    """Fixed temporal samples; no endpoint label, target distance or image input."""
    n = len(trajectory)
    if not n:
        return []
    chosen, seen = [], set()
    for name, index in [('start', 0), ('middle', (n-1)//2), ('end', n-1)]:
        key = tuple(np.asarray(trajectory[index]).round(5))
        if key not in seen:
            chosen.append((name, index)); seen.add(key)
    return chosen


def select_groups(labels, split, count):
    groups = {}
    for row in labels:
        if row['split'] == split:
            groups.setdefault(row['group'], []).append(row['index'])
    rng = np.random.default_rng(42)
    selected = rng.permutation(sorted(groups))[:count]
    return sorted(int(rng.choice(groups[group])) for group in selected)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('data', 'cache', 'data-root', 'sbf-source', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--fit', type=int, default=512)
    p.add_argument('--development', type=int, default=128)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(a.sbf_source))
    from sbfnav.dataset import CityNavDataset
    from sbfnav.mapdata import GROUND_LEVEL
    ds = CityNavDataset(a.data_root, 'train_seen', version='processed_citynav')
    labels = json.loads((a.data/'labels_private.json').read_text())
    selected = select_groups(labels, 'fit', a.fit) + select_groups(labels, 'development', a.development)
    assert len({labels[i]['group'] for i in selected}) == len(selected)
    config = dict(selected=sorted(selected), seed=42,
                  sampling='one episode per physical group; start/middle/end reference indices, no GT selection',
                  annotation_sha256=file_sha256(ds.annotation_path),
                  cache_sha256=file_sha256(a.cache/'manifest.json'),
                  code_sha256=file_sha256(__file__),
                  scope='TRAIN reference/teacher observations only; not observed frames of deployed V2 or a closed-loop evaluation')
    mp = a.output/'manifest.json'
    if mp.exists() and json.loads(mp.read_text()) != config:
        raise ValueError('Refuse changed audit inputs')
    atomic_json(config, mp)
    env = CurrentCandidateROIEnvironment(a.data_root/'rgbd', a.output/'observations')
    (a.output/'rows').mkdir(exist_ok=True); (a.output/'views').mkdir(exist_ok=True)
    start = time.perf_counter()
    for ordinal, i in enumerate(sorted(selected)):
        path = a.output/'rows'/f'{i:05d}.json'
        if path.exists():
            continue
        episode = ds[i]
        with np.load(a.cache/'public_features'/f'{i:05d}.npz') as z:
            centers = z['candidate_world'].copy()
        frames = []
        for name, step in reference_indices(episode.trajectory):
            pose = episode.trajectory[step].copy()
            frame = dict(phase=name, step=step, pose=pose.tolist(), unknown_reason=None)
            if not np.isfinite(pose).all() or pose[2] <= GROUND_LEVEL[episode.map_name]:
                frame['unknown_reason'] = 'invalid_or_nonpositive_reference_altitude'
                frames.append(frame); continue
            obs, rois = env.current(episode.map_name, pose, GROUND_LEVEL[episode.map_name], centers)
            valid = np.zeros((len(centers), 2), dtype=bool)
            for roi in rois:
                valid[roi['candidate'], roi['scale']] = roi['valid']
            # Geometry and sensor decisions sealed; private GT now scores proxies.
            distance = np.linalg.norm(centers-labels[i]['target_world'], axis=1)
            known = valid.any(-1)
            positive = known & (distance<=20)
            negative = known & (distance>=40)
            supported_containing_center = 0
            for roi in rois:
                if roi['valid'] and distance[roi['candidate']]<=20:
                    supported_containing_center += int(cv2.pointPolygonTest(
                        np.asarray(roi['quad'], np.float32), tuple(labels[i]['target_world']), False)>=0)
            frame.update(supported=int(known.sum()), positive=int(positive.sum()),
                         negative=int(negative.sum()), weak_trainable=bool(positive.any() and negative.any()),
                         positive_scales=int(valid[positive].sum()),
                         positive_scales_containing_gt_center=supported_containing_center)
            if ordinal < 8:
                image = obs['bgr'].copy()
                from graphnav.landmark_projection import world_to_frame
                points = world_to_frame(centers, obs['footprint_world'], image.shape)
                for j, (x, y) in enumerate(points):
                    if np.isfinite([x,y]).all() and 0<=x<384 and 0<=y<384:
                        cv2.circle(image, (round(float(x)),round(float(y))), 4, (0,255,255), 1)
                        cv2.putText(image, str(j), (round(float(x)),round(float(y))), 0, .35, (0,255,255), 1)
                cv2.imwrite(str(a.output/'views'/f'{i:05d}_{name}.png'), image)
            frames.append(frame)
        atomic_json(dict(index=i, split=labels[i]['split'], map_name=episode.map_name,
                         trajectory_length=len(episode.trajectory), frames=frames), path)
        if ordinal%32==0:
            print(json.dumps(dict(completed=ordinal+1, expected=len(selected), seconds=time.perf_counter()-start)), flush=True)
    rows = [json.loads((a.output/'rows'/f'{i:05d}.json').read_text()) for i in sorted(selected)]
    result = {}
    for split in ('fit', 'development'):
        rr = [r for r in rows if r['split']==split]
        phases = {}
        for phase in ('start', 'middle', 'end'):
            ff = [f for r in rr for f in r['frames'] if f['phase']==phase]
            phases[phase] = dict(n=len(ff), unknown=sum(f['unknown_reason'] is not None for f in ff),
                                 any_supported=sum(f.get('supported',0)>0 for f in ff),
                                 correct_location_supported=sum(f.get('positive',0)>0 for f in ff),
                                 weak_trainable=sum(f.get('weak_trainable',False) for f in ff))
        result[split] = dict(episodes=len(rr), phases=phases,
                             groups_with_trainable_frame=sum(any(f.get('weak_trainable',False) for f in r['frames']) for r in rr))
    atomic_json(dict(summary=result, seconds=time.perf_counter()-start,
                     caveat='All support/labels are geometry proxies; no verified instance accuracy, no navigation SR'),
                a.output/'summary.json')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
