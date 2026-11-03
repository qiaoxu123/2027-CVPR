"""Frozen original priors + current patch hypotheses for TRAIN position learning.

Private target labels are never read here. Original query records are explicitly
sanitized before constructing A3 inputs. Public map names are bound by text only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import cv2
import numpy as np
import rasterio
import torch
from torch.nn import functional as F

from graphnav.appearance_matching import ObservedSigLIPMatchingEncoder
from graphnav.candidate_geometry import FullCandidateDataset,collate_full,relative_entity_features
from graphnav.checkpointing import atomic_json,atomic_save,file_sha256
from graphnav.full_experiment import new_model,_batch_inputs
from graphnav.local_chain_observation import local_patch_regions
from graphnav.observed_patch_scales import masked_native_pool
from graphnav.token_language import FrozenBERTTokens
from scripts.analyze_instruction_landmark_association import PublicNames


def public_query(record):
    allowed=('map_name','instruction','world_bounds','physical_map_sizes','landmarks','dataset_version','match_mode')
    result={k:record[k] for k in allowed if k in record}
    result['target_xy']=[0.,0.]  # Dataset packing scaffold, never in _batch_inputs.
    return result


def assign_maps(maps, frozen=None):
    if frozen is not None:
        if any(m not in frozen for m in maps):
            raise ValueError('missing frozen map partition')
        if any(frozen[m] not in ('fit','dev','confirm') for m in maps):
            raise ValueError('invalid frozen map partition')
        return {m:frozen[m] for m in maps}
    maps=sorted(set(maps),key=lambda m:hashlib.sha256(('chain-split-0:'+m).encode()).hexdigest())
    if len(maps)<9:raise ValueError('at least nine TRAIN maps for disjoint split required')
    return {m:('fit' if i<len(maps)-6 else 'dev' if i<len(maps)-3 else 'confirm') for i,m in enumerate(maps)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('observations','static-queries','public-catalog','checkpoint','roles','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--backbone',required=True)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--device',default='cuda')
    p.add_argument('--partition-from',type=Path,help='Preserve map_assignment from an existing feature manifest')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('do not overwrite cached evidence')
    observed=json.loads((a.observations/'manifest.json').read_text())
    if not observed['complete'] or observed['split']!='train_seen':raise ValueError('complete TRAIN only')
    frames=[json.loads(line) for line in (a.observations/'public_frames.jsonl').read_text().splitlines()]
    if file_sha256(a.observations/'public_frames.jsonl')!=observed['public_frames.jsonl_sha256']:raise ValueError('observations changed')
    queries={f['task_key']:(f['map_name'],f['instruction']) for f in frames}
    selected=set(queries.values()); static={}
    with a.static_queries.open() as stream:
        for line in stream:
            record=json.loads(line);key=(record['map_name'],record['instruction'])
            if key in selected and key not in static:static[key]=public_query(record)
    if set(static)!=selected:raise ValueError('selected queries missing in verified original TRAIN protocol')
    maps={x[0] for x in selected}
    frozen=json.loads(a.partition_from.read_text())['map_assignment'] if a.partition_from else None
    assignment=assign_maps(maps,frozen)
    a.output.mkdir(parents=True)
    meta=dict(complete=False,split='train_seen',map_assignment=assignment,
              partition_source_sha256=file_sha256(a.partition_from) if a.partition_from else None,
              observation_manifest_sha256=file_sha256(a.observations/'manifest.json'),
              roles_sha256=file_sha256(a.roles),public_catalog_sha256=file_sha256(a.public_catalog),
              checkpoint_sha256=file_sha256(a.checkpoint),static_queries_sha256=file_sha256(a.static_queries),
              code_sha256=file_sha256(__file__), chunks=[],camera_yaw_rad=observed['fixed_camera_yaw_rad'],
              global_grid=28,local_candidate_count=144,window_patch_sizes=[1,3,7],
              observation_source=observed.get('source','reference-trajectory TRAIN observations'),
              warning='Frozen TRAIN observation features, not navigation performance. World-goal weak supervision only.')
    atomic_json(meta,a.output/'manifest.json')
    keys=list(queries)
    query_path=a.output/'public_queries.jsonl'
    query_path.write_text(''.join(json.dumps(static[queries[k]])+'\n' for k in keys))
    source=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    args=SimpleNamespace(**source['a3_args'])
    model=new_model(args,torch.device(a.device)).eval().requires_grad_(False)
    model.load_state_dict({k.removeprefix('a3.'):v for k,v in source['model'].items() if k.startswith('a3.')},strict=True)
    bert=FrozenBERTTokens(args.text_backbone,device=a.device,max_length=getattr(args,'max_text_length',96))
    ds=FullCandidateDataset(query_path,grid_size=args.grid_size,max_landmarks=args.max_landmarks,max_roads=args.max_roads,
                            max_relations=args.max_relations,prior_sigma_m=args.prior_sigma_m,rule_sigma_m=args.rule_sigma_m,deduplicate_queries=False)
    priors={}
    started=time.perf_counter()
    with torch.inference_mode():
        for start in range(0,len(keys),8):
            batch=collate_full([ds[i] for i in range(start,min(start+8,len(ds)))])
            inputs,tokens,mask,names=_batch_inputs(batch,bert,torch.device(a.device),'geometry')
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=a.device.startswith('cuda')):
                field=model(inputs,variant='geometry',tokens=tokens,token_mask=mask,node_emb=names)
            for i,key in enumerate(keys[start:start+8]):priors[key]=field.belief_logits[i].float().cpu().clone()
    atomic_save(priors,a.output/'frozen_priors.pt')
    meta['frozen_priors_sha256']=file_sha256(a.output/'frozen_priors.pt')
    del model,bert,ds
    torch.cuda.empty_cache()
    native=ObservedSigLIPMatchingEncoder(a.backbone,device=a.device,microbatch=a.batch_size)
    roles=json.loads(a.roles.read_text())
    target_text={k:native.texts([v['target_text']]).cpu().half() for k,v in roles.items()}
    context_text={k:native.texts([v['context_text']]).cpu().half() for k,v in roles.items()}
    objects=json.loads(a.public_catalog.read_text());public_names=PublicNames(objects)
    entities={}
    for key,(map_name,text) in queries.items():
        mentions=public_names.match(map_name,text);ids=list(dict.fromkeys(oid for m in mentions for oid in m['public_ids']))[:16]
        entities[key]=[dict(kind='region',geometry_m=np.asarray(objects[map_name][oid]['contour'],np.float32)) for oid in ids]
    for offset in range(0,len(frames),a.batch_size):
        batch=frames[offset:offset+a.batch_size];regions=[];images=[]
        for frame in batch:
            if file_sha256(a.observations/frame['image'])!=frame['image_sha256'] or file_sha256(a.observations/frame['depth'])!=frame['depth_sha256']:raise ValueError('RGB/depth changed')
            image=cv2.imread(str(a.observations/frame['image']),cv2.IMREAD_COLOR)
            with np.load(a.observations/frame['depth']) as depth:
                obs=dict(bgr=image,depth_valid=depth['depth_valid'],footprint_world=frame['footprint_world'])
            regions.append(local_patch_regions(obs));images.append(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        pixels=native.processor(images=images,return_tensors='pt').to(native.device)
        records=[]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=a.device.startswith('cuda')):
            patches=native.model.vision_model(**pixels).last_hidden_state
            for frame,(world,masks,valid,quality),patch in zip(batch,regions,patches):
                features=torch.zeros(144,3,1024,device=native.device)
                supported=masks.any(-1)
                if supported.any():
                    pooled=masked_native_pool(native.model.vision_model.head,patch,torch.from_numpy(masks[supported]).to(native.device))
                    features[torch.from_numpy(supported).to(native.device)]=F.normalize(pooled.float(),dim=-1)
                key=frame['task_key'];bounds=np.array(static[queries[key]]['world_bounds'],np.float32)
                geometry=np.zeros((144,31),np.float32)
                if entities[key]:
                    pair=relative_entity_features(world,entities[key],len(entities[key]))
                    geometry=np.concatenate((pair.mean(1),pair.min(1),pair.max(1),np.full((144,1),len(entities[key])/16,np.float32)),-1)
                axis=(np.arange(28,dtype=np.float32)+.5)/28;gx,gy=np.meshgrid(axis,axis,indexing='xy')
                grid=np.stack((gx.ravel(),gy.ravel()),-1)
                records.append(dict(frame_key=frame['frame_key'],task_key=key,map_name=frame['map_name'],
                    subset=assignment[frame['map_name']],step=frame['step'],prior_logits=priors[key],
                    local_uv=torch.from_numpy((world-bounds[:2])/(bounds[2:]-bounds[:2])),
                    candidate_world=torch.from_numpy(np.concatenate((bounds[:2]+grid*(bounds[2:]-bounds[:2]),world),0)),
                    image_features=features.cpu().half().clone(),target_text=target_text[key][0].clone(),
                    context_text=context_text[key][0].clone(),public_geometry=torch.from_numpy(geometry).half(),
                    quality=torch.from_numpy(quality),local_valid=torch.from_numpy(valid)))
        filename=f'features_{offset//a.batch_size:04d}.pt';atomic_save(dict(records=records),a.output/filename)
        meta['chunks'].append(dict(file=filename,sha256=file_sha256(a.output/filename),frames=len(records)))
        meta.update(encoded_frames=offset+len(batch),seconds=time.perf_counter()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        atomic_json(meta,a.output/'manifest.json')
        if offset%64==0:print(json.dumps({k:meta[k] for k in ('encoded_frames','seconds','peak_allocated_gib')}),flush=True)
    meta.update(complete=True,task_counts={s:sum(assignment[m]==s for m,_ in queries.values()) for s in ('fit','dev','confirm')})
    atomic_json(meta,a.output/'manifest.json')


if __name__=='__main__':main()
