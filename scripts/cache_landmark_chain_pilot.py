"""Encode fixed-yaw observed crops with frozen original SigLIP2; no label reads.

Only public_frames.jsonl plus the corresponding captured RGB/depth and public
raster BOUNDS enter this cache. Full raster pixels are not read by this script.
"""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import rasterio
import torch
from torch.nn import functional as F

from graphnav.appearance_matching import ObservedSigLIPMatchingEncoder
from graphnav.checkpointing import atomic_json, atomic_save, file_sha256
from graphnav.evidence_grounding import appearance_query
from graphnav.observed_patch_scales import ObservedPatchScaleEncoder, observed_region_masks, masked_native_pool


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--observations', type=Path, required=True)
    p.add_argument('--rgbd', type=Path, required=True)
    p.add_argument('--backbone', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--roi-extents', type=float, nargs=3, default=[32.,64.,128.])
    a = p.parse_args()
    if not all(np.isfinite(x) and x > 0 for x in a.roi_extents):
        raise ValueError('positive finite ROI extents required')
    if a.output.exists():
        raise FileExistsError('refusing feature cache overwrite')
    source = a.observations/'public_frames.jsonl'
    manifest = json.loads((a.observations/'manifest.json').read_text())
    if not manifest['complete'] or manifest['split'] != 'train_seen':
        raise ValueError('complete TRAIN observations required')
    if file_sha256(source) != manifest['public_frames.jsonl_sha256']:
        raise ValueError('public observations changed')
    frames = [json.loads(line) for line in source.read_text().splitlines()]
    a.output.mkdir(parents=True)
    meta = dict(complete=False, inputs_public_only=True, backbone=a.backbone,
                observation_manifest_sha256=file_sha256(a.observations/'manifest.json'),
                public_frames_sha256=file_sha256(source), source_code_sha256=file_sha256(__file__),
                camera_yaw_rad=manifest['fixed_camera_yaw_rad'], roi_extents_m=a.roi_extents, chunks=[])
    atomic_json(meta, a.output/'manifest.json')
    native = ObservedSigLIPMatchingEncoder(a.backbone, device=a.device, microbatch=a.batch_size)
    encoder = ObservedPatchScaleEncoder(native)
    bounds_by_map = {}
    # Match the original public grid convention (cell centers).
    axis = (np.arange(28, dtype=np.float32)+.5)/28
    xx, yy = np.meshgrid(axis, axis, indexing='xy')
    grid = np.stack([xx.ravel(), yy.ravel()], -1)
    start = time.perf_counter()
    for offset in range(0, len(frames), a.batch_size):
        batch = frames[offset:offset+a.batch_size]
        observations, worlds = [], []
        for frame in batch:
            image = a.observations/frame['image']
            depth_path = a.observations/frame['depth']
            if file_sha256(image) != frame['image_sha256'] or file_sha256(depth_path) != frame['depth_sha256']:
                raise ValueError('captured image/depth checksum mismatch')
            rgb = cv2.imread(str(image), cv2.IMREAD_COLOR)
            with np.load(depth_path) as depth:
                observations.append(dict(bgr=rgb, depth_valid=depth['depth_valid'],
                                         footprint_world=frame['footprint_world']))
            if frame['map_name'] not in bounds_by_map:
                with rasterio.open(a.rgbd/f"{frame['map_name']}.tif") as raster:
                    bounds_by_map[frame['map_name']] = np.array(raster.bounds, np.float32)
            bounds = bounds_by_map[frame['map_name']]
            worlds.append(bounds[:2]+grid*(bounds[2:]-bounds[:2]))
        if a.roi_extents == [32.,64.,128.]:
            result = encoder.encode(observations, worlds)
        else:
            # Explicit experiment-only scale ablation; legacy encoder is unchanged.
            pixels = native.processor(images=[cv2.cvtColor(o['bgr'],cv2.COLOR_BGR2RGB) for o in observations], return_tensors='pt').to(native.device)
            result = []
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=native.device.type=='cuda'):
                patches = native.model.vision_model(**pixels).last_hidden_state
                for observation, world, patch in zip(observations, worlds, patches):
                    masks, quality = observed_region_masks(observation, world, extents=a.roi_extents)
                    valid = masks.any(-1)
                    values = torch.zeros(*valid.shape, native.dimension, device=native.device)
                    if valid.any():
                        pooled = masked_native_pool(native.model.vision_model.head, patch, torch.from_numpy(masks[valid]).to(native.device))
                        values[torch.from_numpy(valid).to(native.device)] = F.normalize(pooled.float(),dim=-1)
                    result.append((values.cpu().half(),torch.from_numpy(valid),torch.from_numpy(quality)))
        full_text = native.texts([f['instruction'] for f in batch]).cpu().half()
        visual_text = native.texts([appearance_query(f['instruction']) for f in batch]).cpu().half()
        records = []
        for i, (frame, (features, valid, quality), world) in enumerate(zip(batch, result, worlds)):
            ids = valid.any(-1).nonzero().flatten()
            records.append(dict(frame_key=frame['frame_key'], task_key=frame['task_key'],
                                map_name=frame['map_name'], step=frame['step'],
                                native_text=full_text[i].clone(), appearance_text=visual_text[i].clone(),
                                grid_world=torch.from_numpy(world),
                                bounds=torch.from_numpy(bounds_by_map[frame['map_name']].copy()),
                                roi_ids=ids, roi_features=features[ids], roi_valid=valid[ids], roi_quality=quality[ids]))
        filename = f'features_{offset//a.batch_size:04d}.pt'
        atomic_save(dict(records=records), a.output/filename)
        meta['chunks'].append(dict(file=filename, sha256=file_sha256(a.output/filename), frames=len(records)))
        meta.update(encoded_frames=offset+len(batch), seconds=time.perf_counter()-start,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if a.device.startswith('cuda') else None)
        atomic_json(meta, a.output/'manifest.json')
        print(json.dumps({k: meta[k] for k in ('encoded_frames','seconds','peak_allocated_gib')}), flush=True)
    meta.update(complete=True, maps=len(bounds_by_map), backbone_revision=getattr(native.model.config,'_commit_hash',None))
    atomic_json(meta, a.output/'manifest.json')


if __name__ == '__main__':
    main()
