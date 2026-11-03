"""TRAIN annotation-side region labels for a vision-only frozen-feature probe.

Unlike named-reference audit this uses private TRAIN target contours, strictly
as weak loss/evaluation labels. No instance geometry is put in model features.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from graphnav.checkpointing import atomic_json, atomic_save, file_sha256
from graphnav.public_reference_supervision import patch_world_windows
from graphnav.region_semantic_supervision import semantic_region_labels, REGION_CLASSES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('cache', 'observations', 'data-root', 'sbf-source', 'fit-observation-filter', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous training labels')
    sys.path.insert(0, str(args.sbf_source))
    from sbfnav.dataset import CityNavDataset
    dataset = CityNavDataset(args.data_root, 'train_seen', version='processed_citynav')
    manifest = json.loads((args.cache/'manifest.json').read_text())
    if not manifest['complete'] or manifest['split'] != 'train_seen' or manifest['camera_yaw_rad'] != 0:
        raise ValueError('TRAIN fixed-yaw frozen cache only')
    catalog_path = args.data_root/'cityrefer/objects.json'
    if file_sha256(catalog_path) != manifest['public_catalog_sha256']:
        raise ValueError('annotation/catalog alignment changed')
    catalog = json.loads(catalog_path.read_text())
    registry = {m: set() for m, subset in manifest['map_assignment'].items() if subset in ('fit', 'dev')}
    for raw in dataset._raw_records:
        if raw['split'] != 'train_seen':
            raise ValueError('non-TRAIN registry record')
        map_name = f"{raw['area']}_block_{raw['block']}"
        if map_name in registry:
            registry[map_name].add(str(raw['object_ids'][0]))
    contours = {m: {c: [] for c in REGION_CLASSES} for m in registry}
    registry_counts = {subset: Counter() for subset in ('fit', 'dev')}
    for map_name, ids in registry.items():
        for oid in sorted(ids):
            obj = catalog[map_name][oid]; category = obj['object_type']
            if category in REGION_CLASSES:
                contours[map_name][category].append(obj['contour'])
                registry_counts[manifest['map_assignment'][map_name]][category] += 1
    frames_path = args.observations/'public_frames.jsonl'
    obs_manifest = json.loads((args.observations/'manifest.json').read_text())
    if file_sha256(frames_path) != obs_manifest['public_frames.jsonl_sha256']:
        raise ValueError('observation metadata changed')
    frames = {r['frame_key']: r for r in map(json.loads, frames_path.read_text().splitlines())}
    allowed = set(json.loads(args.fit_observation_filter.read_text())['frame_keys'])
    labels = {}; records = []; counts = {s: Counter() for s in ('fit', 'dev')}
    started = time.perf_counter()
    for chunk in manifest['chunks']:
        path = args.cache/chunk['file']
        if file_sha256(path) != chunk['sha256']:
            raise ValueError('feature chunk changed')
        for row in torch.load(path, map_location='cpu', weights_only=True)['records']:
            subset = row['subset']; key = row['frame_key']
            if subset not in ('fit', 'dev') or (subset == 'fit' and key not in allowed):
                continue
            frame = frames[key]
            windows = patch_world_windows(frame['footprint_world'])
            valid = row['local_valid'].numpy() & (row['quality'][:, 0].numpy() >= .9)
            label, _ = semantic_region_labels(windows[:, 0], contours[row['map_name']], valid)
            labels[key] = torch.from_numpy(label)
            counts[subset]['frames'] += 1; counts[subset]['all_patches'] += len(label)
            counts[subset]['unknown_patches'] += int((label == -1).sum())
            for index, category in enumerate(REGION_CLASSES):
                counts[subset][category+'_patches'] += int((label == index).sum())
            records.append(dict(frame_key=key, task_key=row['task_key'], map_name=row['map_name'], subset=subset,
                feature_chunk=chunk['file'], labelled_patches=int((label >= 0).sum()),
                by_class={category: int((label == index).sum()) for index, category in enumerate(REGION_CLASSES)},
                annotation='TRAIN contour geometry proxy; not verified RGB labels'))
    args.output.mkdir(parents=True)
    atomic_save(labels, args.output/'loss_labels_private.pt')
    atomic_json(records, args.output/'records.json')
    summary = dict(complete=True, classes=list(REGION_CLASSES), registry={k: dict(v) for k, v in registry_counts.items()},
        counts={k: dict(v) for k, v in counts.items()}, seconds=time.perf_counter()-started,
        uncertain_margin_m=2., positive_coverage=.9, cross_class_overlap_policy='unknown',
        unlabeled_region_policy='unknown, never background',
        fit_map_count=sum(v == 'fit' for v in manifest['map_assignment'].values()),
        dev_map_count=sum(v == 'dev' for v in manifest['map_assignment'].values()),
        private_target_ids_and_contours_read=True, private_fields_used_as_model_inputs=False,
        warning='Land-use proxy recognition only, not pixel-instance accuracy, visibility or navigation SR')
    for key, path in [('feature_manifest', args.cache/'manifest.json'), ('train_annotations', dataset.annotation_path),
                       ('catalog', catalog_path), ('observations', frames_path), ('filter', args.fit_observation_filter),
                       ('labels', args.output/'loss_labels_private.pt'), ('records', args.output/'records.json'),
                       ('producer', Path(__file__)), ('label_code', Path('graphnav/region_semantic_supervision.py'))]:
        summary[key+'_sha256'] = file_sha256(path)
    atomic_json(summary, args.output/'summary.json')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
