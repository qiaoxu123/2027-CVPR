"""All-TRAIN static fields + privileged whole/local reference crops. No Val/Test."""
import argparse,json,sys,time,hashlib
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from graphnav.checkpointing import atomic_json,atomic_save,file_sha256
from graphnav.candidate_geometry import FullCandidateDataset,collate_full
from graphnav.full_relation_training import load_full_relation_model,full_relation_inputs
from graphnav.token_language import FrozenBERTTokens
from graphnav.appearance_matching import ObservedSigLIPMatchingEncoder
from graphnav.renderer import CityNavObservedViewRenderer
from graphnav.privileged_landmark_rgb import privileged_whole_landmark_rgb
from graphnav.native_dense_regions import native_dense_regions,sparse_view
from graphnav.observed_patch_scales import masked_native_pool
from graphnav.polygon_guided_visual import polygon_shape_tokens
from graphnav.local_landmark_anchor import local_anchor_tokens,local_landmark_anchor


def ortho_view(renderer,map_name,footprint):
    renderer._open(map_name);r=renderer._raster;n=384;fp=np.array(footprint)
    corners=np.array([[r.index(float(x),float(y))[1],r.index(float(x),float(y))[0]] for x,y in fp],np.float32)
    matrix=cv2.getPerspectiveTransform(corners,np.float32([[0,0],[383,0],[383,383],[0,383]]));rgb=cv2.warpPerspective(renderer._rgb,matrix,(n,n),flags=cv2.INTER_LINEAR)
    if renderer._height_valid is None:
        z=r.read(1);renderer._height_valid=(np.isfinite(z)&(z>=0)&(r.read_masks(1)>0)).astype(np.float32)
    valid=(cv2.warpPerspective(renderer._height_valid,matrix,(n,n),flags=cv2.INTER_LINEAR)>=.999)&(rgb.max(-1)>0)
    return dict(bgr=rgb,depth_valid=valid,depth_m=np.full((n,n),np.nan,np.float32),footprint_world=fp,provenance='PRIVILEGED_SOURCE_ORTHOPHOTO_NOT_UAV_OBSERVATION',pseudo_view=True)


@torch.inference_mode()
def encode_views(native,observations,contours,names,types,bounds):
    if not observations:return None,[]
    names=list(dict.fromkeys(names));types=list(dict.fromkeys('Road' if t in ('TrafficRoad','Footpath','Bridge') else t for t in types))
    name=native.texts(names).mean(0).cpu().clone();typ=native.texts(types).mean(0).cpu().clone();result=[];stats=[];b=np.asarray(bounds)
    for start in range(0,len(observations),8):
        chunk=observations[start:start+8];batch=native.processor(images=[cv2.cvtColor(x['bgr'],cv2.COLOR_BGR2RGB) for x in chunk],return_tensors='pt').to(native.device)
        with torch.autocast('cuda',dtype=torch.bfloat16):patches=native.model.vision_model(**batch).last_hidden_state
        for obs,patch in zip(chunk,patches):
            dense=sparse_view(native_dense_regions(obs));valid=np.array(dense['valid'])&(dense['quality'][:,0]>=.9);uv=(dense['world']-b[:2])/(b[2:]-b[:2]);valid&=(uv>=0).all(-1)&(uv<=1).all(-1)
            masks=dense['masks'].reshape(-1,576);keep=masks.any(-1);features=torch.zeros(len(masks),native.dimension,device=native.device)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                if keep.any():features[torch.from_numpy(keep).to(native.device)]=F.normalize(masked_native_pool(native.model.vision_model.head,patch,torch.from_numpy(masks[keep]).to(native.device)).float(),dim=-1)
            anchor=local_landmark_anchor(contours,obs)
            shape=polygon_shape_tokens(contours,obs['footprint_world']) if anchor['local_geometry'].is_empty else local_anchor_tokens(contours,obs['footprint_world'])
            result.append(dict(image_features=features.reshape(144,3,-1).cpu().half().clone(),valid=torch.from_numpy(valid),local_uv=torch.from_numpy(uv).float(),name_text=name,type_text=typ,polygon_tokens=torch.from_numpy(shape)))
            side=float(np.linalg.norm(np.array(obs['footprint_world'])[1]-np.array(obs['footprint_world'])[0]))
            stats.append(dict(source_valid_fraction=float(obs['depth_valid'].mean()),valid_patches=int(valid.sum()),positive_anchor_patches=anchor['positive_patches'],polygon_in_view_fraction=anchor['whole_fraction'],side_m=side,meters_per_pixel=side/384,meters_per_patch=side/24,image_SHA256=hashlib.sha256(obs['bgr'].tobytes()).hexdigest()))
    pack={k:torch.stack([x[k] for x in result]) for k in result[0]}
    return pack,stats


def main():
    p=argparse.ArgumentParser()
    for k in ('inventory','train-jsonl','checkpoint','stage1','bert-cache','data-root','sbf-source','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--limit',type=int);a=p.parse_args();sys.path.insert(0,str(a.sbf_source));from sbfnav.mapdata import GROUND_LEVEL
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'groups').mkdir(exist_ok=True);(a.output/'queries').mkdir(exist_ok=True);torch.set_num_threads(4)
    fp=dict(inventory=file_sha256(a.inventory/'manifest.json'),checkpoint=file_sha256(a.checkpoint),source=file_sha256(a.train_jsonl),privileged=True,code=file_sha256(__file__),plan_code=file_sha256('graphnav/complete_landmark_rgb.py'))
    manifest=a.output/'manifest.json'
    if manifest.exists() and json.load(manifest.open())!=fp:raise ValueError('Immutable cache inputs/code differ')
    atomic_json(fp,manifest);qs=json.load(open(a.inventory/'queries_public.json'));groups=json.load(open(a.inventory/'groups.json'));records=[json.loads(x) for x in a.train_jsonl.read_text().splitlines()]
    raw=json.loads((a.data_root/'processed_citynav/citynav_train_seen.json').read_text());assert len(qs)==len(raw)==len(records)
    source=torch.load(a.checkpoint,weights_only=True,map_location='cpu');model=load_full_relation_model(source).eval().requires_grad_(False);cfg=source['config']['geometry_args']
    bert=FrozenBERTTokens(cfg['text_backbone'],device='cuda',max_length=cfg['max_text_length']);bert.load_cache(a.bert_cache)
    native=ObservedSigLIPMatchingEncoder(torch.load(a.stage1,weights_only=True,map_location='cpu')['protocol']['backbone'],device='cuda');renderer=CityNavObservedViewRenderer(a.data_root/'rgbd',a.output/'render',384)
    ds=FullCandidateDataset(a.train_jsonl,grid_size=28,max_landmarks=cfg['max_landmarks'],max_roads=cfg['max_roads'],max_relations=cfg['max_relations'],prior_sigma_m=cfg['prior_sigma_m'],rule_sigma_m=cfg['rule_sigma_m'],deduplicate_queries=False)
    started=time.perf_counter();stats=Counter();group_example=0
    for start in range(0,min(a.limit or len(qs),len(qs)),16):
        ids=[i for i in range(start,min(start+16,len(qs),a.limit or len(qs))) if not (a.output/'queries'/f'{i:05d}.pt').exists()]
        if not ids:continue
        inp=full_relation_inputs(collate_full([ds[i] for i in ids]),bert,'cuda');model.reset_episode()
        with torch.inference_mode(),torch.autocast('cuda',enabled=False):out=model.forward_full_relation(**inp)
        for j,idx in enumerate(ids):
            q=qs[idx];current=None;current_stats=[];reason='no_valid_public_polygon'
            if q['group']:
                group=groups[q['group']];path=a.output/'groups'/f'{q["group"]}.pt'
                if not path.exists():
                    plans=[group['plan']['whole']]+group['plan']['tiles'];images=[ortho_view(renderer,q['map_name'],v['footprint_world']) for v in plans]
                    pack,viewstats=encode_views(native,images,group['contours'],group['names'],group['types'],group['world_bounds'])
                    atomic_save(dict(observed=pack,stats=viewstats,privileged=True,group=q['group']),path);stats['new_groups']+=1;stats['new_privileged_images']+=len(images)
                    if group_example<12:
                        directory=a.output/'LOCAL_PRIVILEGED_IMAGES'/q['group'];directory.mkdir(parents=True,exist_ok=True)
                        for z in [0,1,len(images)-1]:
                            if z<len(images):cv2.imwrite(str(directory/f'view_{z:02d}.png'),images[z]['bgr'])
                        group_example+=1
                pose=raw[idx]['trajectory'][0];yaw=float(np.arctan2(pose[4],pose[3])) if len(pose)==6 else float(pose[3]);pose=[*pose[:3],yaw]
                if pose[2]>GROUND_LEVEL[q['map_name']]:
                    obs=renderer.current_rgbd(q['map_name'],pose,GROUND_LEVEL[q['map_name']]);anchor=local_landmark_anchor(group['contours'],obs)
                    if anchor['pixel_usable'] and anchor['positive_patches']>=2:
                        current,current_stats=encode_views(native,[obs],group['contours'],group['names'],group['types'],group['world_bounds']);reason='available'
                    else:reason='no_supported_current_anchor'
                else:reason='nonpositive_camera_altitude'
            atomic_save(dict(index=idx,base=out['relation_field'].belief_logits[j].cpu().float().clone(),dense=out['geometry_field'].dense_context[j].cpu().half().clone(),
                group=q['group'],subset=q['subset'],current=current,current_stats=current_stats,current_reason=reason),a.output/'queries'/f'{idx:05d}.pt');stats[reason]+=1
        print(json.dumps(dict(query=ids[-1],total=len(qs),seconds=time.perf_counter()-started,counts=stats)),flush=True)
    files=list((a.output/'queries').glob('*.pt'))
    if not a.limit:assert len(files)==len(qs)
    atomic_json(dict(complete=not bool(a.limit),queries_cached=len(files),expected=len(qs),seconds=time.perf_counter()-started,stats=stats,peak_GPU_GiB=torch.cuda.max_memory_allocated()/2**30),a.output/'complete.json')

if __name__=='__main__':main()
