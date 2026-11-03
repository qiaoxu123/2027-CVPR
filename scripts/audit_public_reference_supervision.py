"""TRAIN public named-region weak-label audit; no target labels or coordinates.

Read only current observation metadata, frozen image features, and literal
public names. Anonymous CityRefer objects are never selected as references.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch
from shapely.geometry import Polygon

from graphnav.checkpointing import atomic_json, atomic_save, file_sha256
from graphnav.public_reference_supervision import patch_world_windows, reference_region_labels
from scripts.analyze_instruction_landmark_association import PublicNames, normalise

SCALES = (1, 3, 7)


def summarize(rows):
    result = dict(reference_frames=len(rows), frames=len({r['frame_key'] for r in rows}),
                  tasks=len({r['task_key'] for r in rows}),
                  reference_groups=len({r['reference_group'] for r in rows}),
                  name_families=len({r['name_family'] for r in rows}), by_scale={})
    for index, scale in enumerate(SCALES):
        active = [r for r in rows if r['scale_counts'][index]['eligible']]
        result['by_scale'][str(scale)] = dict(
            eligible_reference_frames=len(active), frames=len({r['frame_key'] for r in active}),
            tasks=len({r['task_key'] for r in active}),
            reference_groups=len({r['reference_group'] for r in active}),
            name_families=len({r['name_family'] for r in active}),
            positives=sum(r['scale_counts'][index]['positive'] for r in active),
            negatives=sum(r['scale_counts'][index]['negative'] for r in active))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'observations', 'catalog', 'fit-observation-filter', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier weak supervision')
    manifest = json.loads((args.cache/'manifest.json').read_text())
    obs_manifest = json.loads((args.observations/'manifest.json').read_text())
    if not manifest['complete'] or manifest['split'] != 'train_seen' or manifest['camera_yaw_rad'] != 0:
        raise ValueError('complete fixed-yaw TRAIN observations only')
    frames_path = args.observations/'public_frames.jsonl'
    if file_sha256(frames_path) != obs_manifest['public_frames.jsonl_sha256']:
        raise ValueError('public observation metadata changed')
    if manifest['public_catalog_sha256'] != file_sha256(args.catalog):
        raise ValueError('public map differs from frozen feature protocol')
    catalog = json.loads(args.catalog.read_text()); names = PublicNames(catalog)
    frames = {r['frame_key']: r for r in map(json.loads, frames_path.read_text().splitlines())}
    allowed = set(json.loads(args.fit_observation_filter.read_text())['frame_keys'])
    rows = []; tensors = {}; counts = Counter(); text_queries = {}
    started = time.perf_counter()
    for chunk in manifest['chunks']:
        path = args.cache/chunk['file']
        if file_sha256(path) != chunk['sha256']:
            raise ValueError('frozen feature chunk changed')
        for feature in torch.load(path, map_location='cpu', weights_only=True)['records']:
            subset = feature['subset']
            if subset not in ('fit', 'dev') or (subset == 'fit' and feature['frame_key'] not in allowed):
                continue
            if {'target_world', 'target_id', 'instance_labels'} & feature.keys():
                raise ValueError('public feature cache unexpectedly contains private training labels')
            frame = frames[feature['frame_key']]; key = (frame['map_name'], frame['instruction'])
            if key not in text_queries:
                mentions = names.match(*key)
                ids = sorted({m['public_ids'][0] for m in mentions if len(m['public_ids']) == 1})
                ambiguous = sum(len(m['public_ids']) != 1 for m in mentions)
                text_queries[key] = (ids, ambiguous)
            ids, ambiguous = text_queries[key]
            counts[subset+'_observations'] += 1
            counts[subset+'_ambiguous_name_mentions'] += ambiguous
            windows = patch_world_windows(frame['footprint_world'])
            valid = feature['local_valid'].numpy()
            footprint = Polygon(frame['footprint_world'])
            for oid in ids:
                obj = catalog[frame['map_name']][oid]
                if obj['object_type'] not in ('Building', 'Ground', 'Parking', 'TrafficRoad'):
                    counts['unsupported_public_type'] += 1; continue
                contour = np.asarray(obj['contour'], float)
                if contour.ndim != 2 or contour.shape[-1] != 2 or len(contour) < 3 or not np.isfinite(contour).all():
                    counts['invalid_public_polygon'] += 1; continue
                polygon = Polygon(contour)
                if not polygon.is_valid or polygon.area <= 0:
                    counts['invalid_public_polygon'] += 1; continue
                labels = []; coverage = []; scale_counts = []
                for index, scale in enumerate(SCALES):
                    supported = valid & (feature['quality'][:, index].numpy() >= .9)
                    label, cov = reference_region_labels(windows[:, index], contour, supported)
                    labels.append(label); coverage.append(cov)
                    positive, negative = int((label == 1).sum()), int((label == 0).sum())
                    scale_counts.append(dict(positive=positive, negative=negative,
                        unknown=int((label == -1).sum()), eligible=bool(positive and negative)))
                sample_key = frame['frame_key']+'__public_'+oid
                tensors[sample_key] = dict(labels=torch.from_numpy(np.stack(labels)),
                                           coverage=torch.from_numpy(np.stack(coverage)))
                rows.append(dict(sample_key=sample_key, frame_key=frame['frame_key'], task_key=frame['task_key'],
                    subset=subset, map_name=frame['map_name'], instruction=frame['instruction'],
                    public_id=oid, public_name=obj['name'], object_type=obj['object_type'],
                    reference_group=frame['map_name']+'::'+oid, name_family=normalise(obj['name'])[0],
                    feature_chunk=chunk['file'], image=frame['image'], image_sha256=frame['image_sha256'],
                    scale_counts=scale_counts,
                    geometric_observed_fraction=float(polygon.intersection(footprint).area/polygon.area),
                    visibility='unknown', registration='geometry_proxy', pixel_gt=False,
                    uncertainty_flags=['unverified_occlusion', 'unverified_polygon_to_RGB_alignment', 'unverified_height']))
    fit_names = {r['name_family'] for r in rows if r['subset'] == 'fit'}
    summary = dict(complete=True, scopes=['TRAIN-fit', 'TRAIN-dev'], counts=dict(counts),
        by_subset={subset: summarize([r for r in rows if r['subset'] == subset]) for subset in ('fit', 'dev')},
        dev_unseen_name=summarize([r for r in rows if r['subset'] == 'dev' and r['name_family'] not in fit_names]),
        dev_unseen_name_definition='normalized public name absent from every selected fit observation; not guaranteed physical identity uniqueness',
        supervision='geometry proxy only, no confirmed RGB instance or visibility labels',
        private_target_labels_read=False, normal_inference_modified=False,
        positive_coverage=.9, negative_coverage=.01, uncertainty_margin_m=2.,
        scale_patch_sizes=list(SCALES), no_positive_policy='exclude from ranking; never create Null absence labels',
        seconds=time.perf_counter()-started)
    args.output.mkdir(parents=True)
    atomic_save(tensors, args.output/'weak_labels.pt')
    atomic_json(rows, args.output/'records.json')
    for key, path in [('feature_manifest', args.cache/'manifest.json'), ('observation_manifest', args.observations/'manifest.json'),
                       ('public_frames', frames_path), ('catalog', args.catalog), ('filter', args.fit_observation_filter),
                       ('weak_labels', args.output/'weak_labels.pt'), ('records', args.output/'records.json'),
                       ('producer_code', Path(__file__)), ('label_code', Path('graphnav/public_reference_supervision.py'))]:
        summary[key+'_sha256'] = file_sha256(path)
    atomic_json(summary, args.output/'summary.json')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
