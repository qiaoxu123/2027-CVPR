"""Real TRAIN observations; complete original BERT and frozen SigLIP2 cache."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import cv2
import numpy as np
import torch
from graphnav.checkpointing import atomic_save, file_sha256
from graphnav.candidate_geometry import FullCandidateDataset, collate_full
from graphnav.full_experiment import _batch_inputs
from graphnav.direct_target_data import relation_geometry
from graphnav.token_language import FrozenBERTTokens
from graphnav.appearance_matching import ObservedSigLIPMatchingEncoder
from graphnav.renderer import CityNavObservedViewRenderer
from graphnav.pgvb_observation import encode_landmark_observation
from graphnav.landmark_alias import resolve_references


def main():
    p = argparse.ArgumentParser()
    for k in ('inventory', 'checkpoint', 'stage1', 'data-root', 'sbf-source', 'output'):
        p.add_argument('--'+k, type=Path, required=True)
    p.add_argument('--limit', type=int)
    p.add_argument('--reuse-visual-cache', type=Path,
                   help='Rebuild public geometry/BERT only; preserve exact observed visual tensors')
    a = p.parse_args(); sys.path.insert(0, str(a.sbf_source))
    from sbfnav.mapdata import GROUND_LEVEL
    torch.set_num_threads(4)
    tasks = json.loads((a.inventory/'public_tasks.json').read_text())
    bundle = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    stage = torch.load(a.stage1, map_location='cpu', weights_only=True)
    config = SimpleNamespace(**bundle['config']['geometry_args'])
    a.output.mkdir(parents=True, exist_ok=True)
    fingerprint = dict(public_tasks_sha256=file_sha256(a.inventory/'public_tasks.json'),
                       checkpoint_sha256=file_sha256(a.checkpoint),
                       code_sha256={k:file_sha256(k) for k in [__file__,
                          'graphnav/direct_target_data.py', 'graphnav/pgvb_observation.py']},
                       observation_protocol='TRAIN start/middle/last actual pose; no GT-based pose selection',
                       labels_separate=True, foundation_encoders_frozen=True)
    if a.reuse_visual_cache:
        fingerprint['visual_reuse_manifest_sha256'] = file_sha256(a.reuse_visual_cache/'manifest.json')
    mp = a.output/'manifest.json'
    if mp.exists() and json.loads(mp.read_text()) != fingerprint:
        raise ValueError('Cache inputs/code changed; use new directory')
    mp.write_text(json.dumps(fingerprint, indent=2))
    records = []
    catalog = json.loads((a.data_root/'cityrefer/objects.json').read_text())
    for task in tasks:
        bounds = np.array(task['world_bounds']); span = bounds[2:]-bounds[:2]
        resolved = resolve_references([q['landmark']['name'] for q in task['anchors']],
            catalog[task['map_name']], max_landmarks=config.max_landmarks, match_mode='fuzzy')
        landmarks = [dict(name=q['name'], object_type=q['object_type'], aliases=q['aliases'],
                          contour=((np.asarray(q['contour'])-bounds[:2])/span).tolist()) for q in resolved]
        records.append(dict(map_name=task['map_name'], instruction=task['instruction'],
                            episode_id=str(task['episode']), landmarks=landmarks,
                            target_xy=[0., 0.], world_bounds=bounds.tolist(),
                            physical_map_sizes=span.tolist(), dataset_version='revised', match_mode='fuzzy'))
    path = a.output/'public_static.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    dataset = FullCandidateDataset(path, grid_size=28, max_landmarks=config.max_landmarks,
        max_roads=config.max_roads, max_relations=config.max_relations,
        prior_sigma_m=config.prior_sigma_m, rule_sigma_m=config.rule_sigma_m)
    if len(dataset) != len(tasks):
        raise ValueError('Unexpected duplicated public queries; resolve before fitting')
    bert = FrozenBERTTokens(config.text_backbone, device='cuda', max_length=config.max_text_length)
    native = ObservedSigLIPMatchingEncoder(stage['protocol']['backbone'], device='cuda')
    if native.model.config._commit_hash != stage['protocol']['backbone_revision']:
        raise ValueError('SigLIP revision mismatch')
    renderer = CityNavObservedViewRenderer(a.data_root/'rgbd', a.output/'render', 384)
    start = time.perf_counter(); frames=0; useful=0
    for task in tasks[:a.limit]:
        out = a.output/f"{task['index']:05d}.pt"
        if out.exists(): continue
        batch = collate_full([dataset[task['index']]])
        static, tokens, mask, nodes = _batch_inputs(batch, bert, 'cuda', 'geometry')
        semantic = ' '.join([task['structured']['target']['category']] +
                           [x['value'] for x in task['structured']['target']['attributes']])
        with torch.inference_mode(): target_text = native.texts([semantic])[0].cpu().clone()
        query = dict(static={k:v[0].cpu().clone() for k,v in static.items()},
            tokens=tokens[0].cpu().clone(), mask=mask[0].cpu().clone(),
            nodes=nodes[0].cpu().clone(), target_text=target_text,
            relation_geometry=relation_geometry(task, static['grid_xy'][0].cpu().numpy()),
            relation_ids=torch.tensor([x['predicate_id'] for x in task['anchors']]))
        if a.reuse_visual_cache:
            old_manifest = json.loads((a.reuse_visual_cache/'manifest.json').read_text())
            for key in ('public_tasks_sha256', 'checkpoint_sha256'):
                if old_manifest[key] != fingerprint[key]:
                    raise ValueError('Visual reuse input/checkpoint mismatch')
            old = torch.load(a.reuse_visual_cache/f"{task['index']:05d}.pt", map_location='cpu', weights_only=True)
            if old['index'] != task['index'] or old['subset'] != task['subset']:
                raise ValueError('Visual reuse query mismatch')
            observations = old['observations']
            for row in observations:
                if row.get('RGB'):
                    os.link(a.reuse_visual_cache/row['RGB'], a.output/row['RGB'])
            atomic_save(dict(index=task['index'],subset=task['subset'],query=query,observations=observations),out)
            if task['index'] % 50 == 0:
                print(json.dumps(dict(query=task['index'],mode='geometry_corrected_exact_visual_reuse',seconds=time.perf_counter()-start)),flush=True)
            continue
        observations = []
        anchor = task['anchors'][0]
        landmark = dict(name=anchor['landmark']['name'], typ=anchor['landmark']['type'], contours=anchor['contours'])
        for source in task['observations']:
            if source['pose'][2] <= GROUND_LEVEL[task['map_name']]:
                observations.append(dict(observed=None, step=source['step'], pose=source['pose'],
                    footprint_world=None, rgb_sha256=None, RGB=None,
                    reason='nonpositive_original_camera_altitude', anchor_labels=None))
                frames += 1
                continue
            obs = renderer.current_rgbd(task['map_name'], source['pose'], GROUND_LEVEL[task['map_name']])
            with torch.inference_mode():
                observed, reason, meta = encode_landmark_observation(obs, landmark,
                    task, native, native.device, return_metadata=True, allow_offscreen_anchor=True)
            rgbsha = hashlib.sha256(obs['bgr'].tobytes()).hexdigest()
            saved = None if observed is None else {k:v[0].cpu().clone() for k,v in observed.items()}
            if saved is not None:
                saved['image_features'] = saved['image_features'].half()
            fname = f"{task['index']:05d}_{source['step']:04d}.png"
            cv2.imwrite(str(a.output/fname), obs['bgr'])
            labels = meta['anchor_labels'][0].cpu() if meta else None
            observations.append(dict(observed=saved, step=source['step'], pose=source['pose'],
                footprint_world=np.asarray(obs['footprint_world']).tolist(), rgb_sha256=rgbsha,
                RGB=fname, reason=reason, anchor_labels=labels))
            frames += 1; useful += int(observed is not None)
        atomic_save(dict(index=task['index'], subset=task['subset'], query=query,
                         observations=observations), out)
        if task['index'] % 20 == 0:
            print(json.dumps(dict(query=task['index'], total=len(tasks), frames_this_run=frames,
                 available_RGB_this_run=useful, seconds=time.perf_counter()-start)), flush=True)
    bert.save_cache(a.output/'bert.pt')
    print(json.dumps(dict(complete=len(list(a.output.glob('[0-9]*.pt'))), expected=len(tasks),
         seconds=time.perf_counter()-start, peak_gpu_GiB=torch.cuda.max_memory_allocated()/2**30)), flush=True)


if __name__ == '__main__': main()
