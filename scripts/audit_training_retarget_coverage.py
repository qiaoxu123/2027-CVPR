"""Count TRAIN-only query re-pairing proposals in existing frozen observations.

No images/features/language priors are modified. Proposals are geometric, not
verified visible-instance matches or a generated training dataset yet.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch

from graphnav.checkpointing import atomic_json, file_sha256
from graphnav.training_retarget_proposals import registered_core_proposals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'fit-observation-filter', 'data-root', 'sbf-source', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Preserve earlier proposal audit')
    manifest = json.loads((args.cache/'manifest.json').read_text())
    if manifest['split'] != 'train_seen' or not manifest['complete']:
        raise ValueError('complete TRAIN features only')
    sys.path.insert(0, str(args.sbf_source))
    from sbfnav.dataset import CityNavDataset
    dataset = CityNavDataset(args.data_root, 'train_seen', version='processed_citynav')
    catalog_path = args.data_root/'cityrefer/objects.json'
    if file_sha256(catalog_path) != manifest['public_catalog_sha256']:
        raise ValueError('registry differs from observation protocol')
    catalog = json.loads(catalog_path.read_text())
    query_indexes = {name: defaultdict(list) for name, subset in manifest['map_assignment'].items() if subset == 'fit'}
    for index, raw in enumerate(dataset._raw_records):
        if raw['split'] != 'train_seen': raise ValueError('non-TRAIN annotation')
        name = f"{raw['area']}_block_{raw['block']}"; oid = str(raw['object_ids'][0])
        if name in query_indexes and catalog[name][oid]['object_type'] == 'Car':
            query_indexes[name][oid].append(index)
    registry = {}
    for name, table in query_indexes.items():
        ids = sorted(table)
        registry[name] = (ids, np.array([catalog[name][oid]['position'][:2] for oid in ids], float).reshape(-1, 2))
    allowed = set(json.loads(args.fit_observation_filter.read_text())['frame_keys'])
    counts = Counter(); proposals = []; observed = set(); eligible = set()
    for chunk in manifest['chunks']:
        path = args.cache/chunk['file']
        if file_sha256(path) != chunk['sha256']: raise ValueError('feature chunk changed')
        for row in torch.load(path, map_location='cpu', weights_only=True)['records']:
            if row['subset'] != 'fit' or row['frame_key'] not in allowed: continue
            name = row['map_name']; ids, positions = registry[name]
            found = registered_core_proposals(row['candidate_world'][784:].numpy(), row['local_valid'].numpy(), positions, ids)
            original_index = int(row['task_key'].removeprefix('train_'))
            original_id = str(dataset._raw_records[original_index]['object_ids'][0])
            original_is_car = catalog[name][original_id]['object_type'] == 'Car'
            counts['fit_observations'] += 1
            counts['observations_with_any_car_core'] += bool(found)
            counts['observations_with_multiple_car_cores'] += len(found) > 1
            if original_is_car:
                counts['original_car_query_observations'] += 1
                counts['original_car_query_has_target_core'] += original_id in found
            for oid, candidates in found.items():
                group = (name, oid); observed.add(group)
                has_negative = len(found) > 1
                if has_negative: eligible.add(group)
                counts['image_instance_pairs'] += 1
                counts['pairs_with_other_instance_core_negatives'] += has_negative
                counts['different_from_original_target_pairs'] += oid != original_id
                proposals.append(dict(frame_key=row['frame_key'], original_task_key=row['task_key'], map_name=name,
                    private_training_instance_id=oid, candidate_indices=candidates,
                    available_train_query_indices=query_indexes[name][oid],
                    has_other_instance_core_negatives=has_negative,
                    same_as_original_target=oid == original_id,
                    visibility='unknown', registration='geometry_proxy',
                    source='existing real TRAIN observation, offline teacher proposal only'))
    args.output.mkdir(parents=True)
    atomic_json(proposals, args.output/'links_private.json')
    summary = dict(counts=dict(counts), distinct_observed_map_instance_groups=len(observed),
        distinct_groups_with_other_instance_negatives=len(eligible),
        teacher_only=True, actual_rgb_visibility_verified=False, normal_inference_modified=False,
        new_instruction_features_computed=False, generated_training_samples=False,
        radius_m=2., neighbor_margin_m=.75,
        warning='World core coverage only, not pixel-instance matching. Re-paired instructions require fresh BERT/A0/SigLIP text and name geometry; never reuse original query priors.')
    for name, path in [('feature_manifest', args.cache/'manifest.json'), ('train_annotations', dataset.annotation_path),
                       ('catalog', catalog_path), ('filter', args.fit_observation_filter),
                       ('links', args.output/'links_private.json'), ('producer', Path(__file__))]:
        summary[name+'_sha256'] = file_sha256(path)
    atomic_json(summary, args.output/'summary.json'); print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__': main()
