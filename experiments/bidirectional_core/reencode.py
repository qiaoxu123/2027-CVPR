"""Numerical precision repair over exactly frozen observed ROI selections.

No candidate, sample, query, label, bbox, camera or language is changed. The
failed BF16 cache remains intact; immutable static/text files are hard-linked.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from graphnav.checkpointing import atomic_json,atomic_save,file_sha256
from graphnav.renderer import CityNavObservedViewRenderer
from graphnav.observed_patch_scales import masked_native_pool
from .cache import CachedCoreDataset,jsonl,DEFAULT_DATA
from .vision import FrozenROIEncoder


@torch.inference_mode()
def fixed_encode(encoder,observations,selections):
    import cv2
    if len(observations)!=len(selections):raise ValueError('fixed ROI frame alignment')
    tensors=encoder.native.processor(images=[cv2.cvtColor(o['bgr'],cv2.COLOR_BGR2RGB) for o in observations],return_tensors='pt').to(encoder.device)
    with torch.autocast('cuda',dtype=torch.float16):
        patches=encoder.native.model.vision_model(**tensors).last_hidden_state
    result=[]
    positions=(np.arange(24)+.5)*16-.5;xx,yy=np.meshgrid(positions,positions)
    for observation,patch,regions in zip(observations,patches,selections):
        if observation['bgr'].shape!=(384,384,3):raise ValueError('real native frame required')
        support=(observation['bgr'].max(-1)>0).reshape(24,16,24,16).mean((1,3)).ravel()>=.9
        masks=[]
        for r in regions:
            x0,y0,x1,y1=r['bbox'];mask=((xx>=x0)&(xx<x1)&(yy>=y0)&(yy<y1)).ravel() & support
            if not mask.any():raise ValueError('frozen selected ROI lost its actual pixel mask')
            masks.append(mask)
        with torch.autocast('cuda',enabled=False):
            pooled=masked_native_pool(encoder.native.model.vision_model.head,patch.float(),torch.tensor(np.stack(masks),device=encoder.device))
            features=F.normalize(pooled.float(),dim=-1).cpu().half()
        if not torch.isfinite(features).all() or (features.float().norm(dim=-1)-1).abs().max()>.002:
            raise ValueError('repaired ROI finite/unit-normalization gate failed')
        result.append(features)
    return result


def reencode(args):
    start=time.perf_counter();root=Path(args.output);source=Path(args.source)
    if root.exists():raise ValueError('refusing numerical repair overwrite')
    old=CachedCoreDataset(source);manifest=copy.deepcopy(old.manifest)
    for p,sha in manifest['frozen_resources'].items():
        if file_sha256(p)!=sha:raise ValueError('source resource changed before precision repair')
    prefixpaths=[p for p in manifest['frozen_resources'] if Path(p).name=='pilot_prefixes.jsonl']
    if len(prefixpaths)!=1:raise ValueError('missing frozen prefix source')
    prefixes={r['sample_id']:r for r in jsonl(prefixpaths[0])}
    root.mkdir(parents=True);manifest.update(complete=False,files={},chunks=[],image_precision='fp16_backbone_fp32_head')
    manifest['precision_repair']=dict(source=str(source.resolve()),source_manifest_sha256=file_sha256(source/'manifest.json'),
        reason='BF16 masked native ROI minimum image cosine below fixed .995 gate; BF16vision+FP32head also failed',
        vision_dtype='float16',region_head_dtype='float32',storage_dtype='float16',
        sample_ids_unchanged=True,selection_unchanged=True,labels_unchanged=True,queries_unchanged=True,
        code_sha256=file_sha256(__file__),vision_code_sha256=file_sha256(Path(__file__).with_name('vision.py')))
    for name in ('queries.pt','queries.jsonl'):
        os.link(source/name,root/name);manifest['files'][name]=file_sha256(root/name)
    atomic_json(manifest,root/'manifest.json')
    encoder=FrozenROIEncoder(device=args.device,max_rois=32)
    if encoder.image_precision!='fp16_backbone_fp32_head':raise ValueError('numerical precision mode is not frozen')
    renderer=CityNavObservedViewRenderer(Path(args.data_root)/'rgbd',root/'environment_private',image_size=384)
    nonempty=[i for i,r in enumerate(old.rows) if len(r['public']['roi_features'])]
    maxnorm=0.;equivalence=None;totalroi=0
    for begin in range(0,len(nonempty),args.batch_size):
        indices=nonempty[begin:begin+args.batch_size];observations=[];selections=[]
        for i in indices:
            row=old.rows[i];m=row['metadata'];p=prefixes[m['sample_id']]
            if p['cutoff_step']!=m['cutoff_step'] or p['source_step']!=m['source_step']:
                raise ValueError('frozen source-direction future/cutoff mismatch')
            if p['pose_world']!=m['actual_camera_pose'] or p['pose_world'][2]-p['ground_level']<=0:
                raise ValueError('actual camera/AGL provenance mismatch')
            observation=renderer.current_rgbd(m['map_name'],np.asarray(p['pose_world']),p['ground_level'])
            from graphnav.memory import ObservedRGBMemory
            expected=ObservedRGBMemory.view_corners(np.asarray(p['pose_world'])[:2],p['pose_world'][3],p['pose_world'][2]-p['ground_level'])
            if not np.array_equal(observation['pose'],np.asarray(p['pose_world'])) or not np.array_equal(observation['footprint_world'],expected):
                raise ValueError('actual current frame footprint/pose mismatch')
            observations.append(observation)
            selections.append(m['roi_objects'])
        vectors=fixed_encode(encoder,observations,selections)
        # Official shared encoder route comparison using the same first real frame
        # catches precision/head/preprocessing drift without changing selection.
        if equivalence is None:
            from .cache import public_landmarks
            objects=json.loads((Path(args.data_root)/'cityrefer/objects.json').read_text())
            lm=public_landmarks(objects);m=old.rows[indices[0]]['metadata']
            native=encoder.encode(observations,[lm[old.rows[i]['metadata']['map_name']] for i in indices])
            errors=[];cosines=[]
            for (regions,official),selection,repaired in zip(native,selections,vectors):
                if [r['object_id'] for r in regions]!=[r['object_id'] for r in selection] or [r['bbox'] for r in regions]!=[r['bbox'] for r in selection]:
                    raise ValueError('fixed region selection differs from original official route')
                errors.append(float((official.float()-repaired.float()).abs().max()))
                cosines.append(float(F.cosine_similarity(official.float(),repaired.float()).min()))
            # Compare implementations with identical batch grouping. A separately
            # recorded single-frame check quantifies expected AMP batch rounding.
            single=encoder.encode([observations[0]],[lm[m['map_name']]])[0][1]
            equivalence=dict(max_abs_error=max(errors),min_cosine=min(cosines),same_batch_frames=len(indices),
                single_vs_batch_max_abs_error=float((single.float()-vectors[0].float()).abs().max()),
                single_vs_batch_min_cosine=float(F.cosine_similarity(single.float(),vectors[0].float()).min()))
            atomic_json(equivalence,root/'route_equivalence.json');print(json.dumps(equivalence),flush=True)
            if equivalence['min_cosine']<.999 or equivalence['max_abs_error']>.001:
                raise ValueError('fixed-mask and shared runtime native same-batch region route mismatch')
        for i,v in zip(indices,vectors):
            old.rows[i]['public']['roi_features']=v;totalroi+=len(v)
            maxnorm=max(maxnorm,float((v.float().norm(dim=-1)-1).abs().max()))
        if begin%256==0:
            print(json.dumps(dict(completed_nonempty=min(begin+len(indices),len(nonempty)),total_nonempty=len(nonempty),rois=totalroi,seconds=time.perf_counter()-start)),flush=True)
    # Preserve original chunk row counts, metadata, labels and candidate ordering.
    cursor=0
    for name in old.manifest['chunks']:
        previous=torch.load(source/name,map_location='cpu',weights_only=True);count=len(previous);rows=old.rows[cursor:cursor+count]
        for a,b in zip(previous,rows):
            if a['metadata']!=b['metadata'] or a['query_index']!=b['query_index']:raise ValueError('precision repair changed sample metadata')
            for key in a['labels']:
                if not torch.equal(a['labels'][key],b['labels'][key]):raise ValueError('precision repair changed labels')
            for key in a['public']:
                if key!='roi_features' and not torch.equal(a['public'][key],b['public'][key]):raise ValueError('precision repair changed nonimage inputs')
        atomic_save(rows,root/name);manifest['chunks'].append(name);manifest['files'][name]=file_sha256(root/name);cursor+=count
    if cursor!=len(old.rows):raise ValueError('repair truncation')
    for p,sha in manifest['frozen_resources'].items():
        if file_sha256(p)!=sha:raise ValueError('resource mutation during repair')
    manifest.update(complete=True,precision_repair_seconds=time.perf_counter()-start)
    manifest['precision_repair'].update(nonempty_states=len(nonempty),encoded_rois=totalroi,norm_max_abs_error=maxnorm,
        runtime_route_equivalence=equivalence,all_other_inputs_labels_equal=True,frozen_resources_unchanged=True)
    atomic_json(manifest,root/'manifest.json');print(json.dumps(manifest['precision_repair']),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--data-root',default=DEFAULT_DATA);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--device',default='cuda');reencode(p.parse_args())
if __name__=='__main__':main()
