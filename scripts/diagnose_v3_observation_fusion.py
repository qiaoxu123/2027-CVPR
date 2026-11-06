"""TRAIN-only frozen paired diagnosis. Does not change model/runtime/training code."""
import argparse, hashlib, json, sys, time
from collections import OrderedDict
from pathlib import Path
import cv2
import numpy as np
import torch
from graphnav.appearance_matching import ObservedSigLIPMatchingEncoder
from graphnav.checkpointing import atomic_json, file_sha256
from graphnav.renderer import CityNavObservedViewRenderer
from graphnav.unified_pgvb_v3 import UnifiedPGVBV3, project_global_to_frame, world_grid
from scripts.unified_pgvb_v3_pilot import V3CacheDataset, fit_dev_partition, _collate, _load_base, _depth_patch_tensor, blind_public_landmarks

VARIANTS=('start_native','start_yaw0','start_h80_yaw0','plan1_yaw0','plan1_nativeyaw','plan5_yaw0')

def top_position(logits,bounds):
    idx=int(logits.flatten().topk(20).indices[0]); n=logits.shape[-1]
    return bounds[:2]+torch.tensor([(idx%n+.5)/n,(idx//n+.5)/n],device=logits.device)*(bounds[2:]-bounds[:2])

def score(logits,target,bounds):
    ne=float((top_position(logits,bounds)-target).norm())
    return dict(NE=ne,hit=ne<=20.)

def candidate_changes(output,gain,bounds):
    """All rules consume public predictions only; target absent by signature."""
    g=output.geometry_global[0];v=output.visual_global[0];m=output.observed_mask[0]
    delta=gain*torch.tanh(v/5)*m
    base=top_position(g,bounds)
    xy=bounds[:2]+world_grid(384,device=g.device,dtype=g.dtype)*(bounds[2:]-bounds[:2])
    away=(xy-base).norm(dim=-1)>20
    margin=float(g.max()-g.masked_fill(~away,float('-inf')).max())
    factor=min(4.,max(1.,margin/(2*max(abs(gain),1e-6))))
    if m.any():
        visual_peak=top_position(v.masked_fill(~m,float('-inf')),bounds)
        distant=(xy-visual_peak).norm(dim=-1)>20
        others=m&distant
        gap=float(v[m].max()-v.masked_fill(~others,float('-inf')).max()) if others.any() else 0.
        center=v[m].mean()
    else:gap=0.;center=0.
    return dict(scale1=g+delta,scale2=g+2*delta,scale4=g+4*delta,
                margin_adaptive=g+factor*delta,
                margin_confidence=g+(factor*delta if gap>=1. else 0),
                centered=g+gain*torch.tanh((v-center)/5)*m),dict(geometry_public_margin=margin,adaptive_scale=factor,visual_public_gap=gap)

def support_audit(output,kw,target,bounds,has_anchor):
    """PRIVATE posthoc geometry/support audit, never a model input."""
    xy=bounds[:2]+world_grid(384,device=bounds.device,dtype=bounds.dtype)*(bounds[2:]-bounds[:2])
    near=(xy-target).norm(dim=-1)<=20
    grid,inside=project_global_to_frame(kw['world_bounds'],kw['footprints_world'])
    raw=kw.get('_raw_valid',kw['patch_valid'])
    n=raw.shape[1]
    depth=torch.nn.functional.grid_sample(raw.float().reshape(n,1,24,24),grid.reshape(n,384,384,2),mode='nearest',align_corners=False).reshape(1,n,384,384)>.5
    geo=inside.any(1)[0];effective=output.observed_mask[0]
    raw_support=(inside&depth).any(1)[0]
    ix=((target-bounds[:2])/(bounds[2:]-bounds[:2])*384).long().clamp(0,383)
    region=bool((near&effective).any()); g=output.geometry_global[0];v=output.visual_global[0];f=output.fused_logits[0]
    good=g.masked_fill(~near,float('-inf')).max();bad=g.masked_fill(near,float('-inf')).max()
    observed=v[effective]
    return dict(anchor=has_anchor,center_in_footprint=bool(geo[ix[1],ix[0]]),
                region_in_footprint=bool((near&geo).any()),raw_depth_region_support=bool((near&raw_support).any()),
                effective_region_support=region,observed_fraction=float(effective.float().mean()),
                raw_patch_valid_n=int(raw.sum()),effective_patch_valid_n=int(kw['patch_valid'].sum()),
                rgb_delta_nonzero=bool(((f-g).abs()>1e-7).any()),
                wrong_minus_correct_peak=float(bad-good),visual_positive_fraction=float((observed>0).float().mean()) if len(observed) else None,
                tanh_saturated_fraction=float((torch.tanh(observed/5).abs()>.95).float().mean()) if len(observed) else None,
                visual_quantiles=torch.quantile(observed,torch.tensor([0.,.1,.5,.9,1.],device=v.device)).tolist() if len(observed) else None)

@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    for key in ('run','source-v2','data-root','sbf-source','train-annotations','objects','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--limit',type=int,default=0,help='TRAIN engineering smoke only; zero=all 4405')
    a=p.parse_args();torch.manual_seed(42);np.random.seed(42);sys.path.insert(0,str(a.sbf_source))
    from sbfnav.dataset import CityNavDataset
    from sbfnav.mapdata import GROUND_LEVEL
    cache=V3CacheDataset(a.run/'cache');fit,heldout=fit_dev_partition(cache,42)
    if not a.limit:assert len(heldout)==4405
    chosen=heldout[:a.limit] if a.limit else heldout
    official=CityNavDataset(a.data_root,'train_seen',version='processed_citynav')
    lookup={':'.join(map(str,official[i].episode_id)):official[i] for i in range(len(official))}
    annotations={str(x['episode_id']):x for x in map(json.loads,a.train_annotations.read_text().splitlines())}
    objects=json.loads(a.objects.read_text());public=[];meta=[]
    for index in chosen:
        lab=torch.load(cache.root/'private'/f'{cache.keys[index]}.pt',weights_only=True)
        identity=lab['episode_id'];rec=lookup[':'.join(identity.split(':')[:3])];ann=annotations[identity]
        public.append(dict(map_name=rec.map_name,instruction=rec.instruction,world_bounds=ann['world_bounds'],landmarks=[]));meta.append((index,identity,rec))
    geom,cfg=_load_base(a.source_v2,'cuda');model=UnifiedPGVBV3(geom).cuda().eval().requires_grad_(False)
    ck=a.run/'train/last.pt';model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True)['model'],strict=True)
    blind_public_landmarks(public,objects,cfg.max_landmarks)
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'rows').mkdir(exist_ok=True)
    manifest=dict(base_commit='1f545e00',checkpoint_sha256=file_sha256(ck),cache_sha256=file_sha256(cache.root/'manifest.json'),
                  dataset='Revised TRAIN heldout only',n=len(chosen),selection_sha256=hashlib.sha256(json.dumps([x[1] for x in meta]).encode()).hexdigest(),
                  variants=VARIANTS,counterfactuals=['scale1','scale2','scale4','margin_adaptive','margin_confidence','centered'],
                  script_sha256=file_sha256(Path(__file__)),no_training=True,no_val_test=True,
                  note='Planned views are offline augmentation, not causal flight history; no target-selected pose. No instance/visibility GT.')
    path=a.output/'manifest.json'
    if path.exists():assert json.loads(path.read_text())==json.loads(json.dumps(manifest)),'immutable run mismatch'
    atomic_json(manifest,path)
    encoder=ObservedSigLIPMatchingEncoder(device='cuda',microbatch=1)
    renderer=CityNavObservedViewRenderer(a.data_root/'rgbd',a.output/'unused',image_size=384)
    dense_cache=OrderedDict()
    def frame(map_name,pose):
        obs=renderer.current_rgbd(map_name,pose,GROUND_LEVEL[map_name]);depth,valid=_depth_patch_tensor(obs)
        key=encoder.image_key(obs['bgr'])
        if key not in dense_cache:
            image=encoder.processor(images=[cv2.cvtColor(obs['bgr'],cv2.COLOR_BGR2RGB)],return_tensors='pt').to('cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):tokens=encoder.model.vision_model(**image).last_hidden_state[0]
            dense_cache[key]=tokens.half().float()
            if len(dense_cache)>128:dense_cache.popitem(last=False)
        return dict(rgb_patches=dense_cache[key][None,None],depth_patches=depth.cuda().half().float()[None,None],
                    patch_valid=valid.cuda()[None,None],footprints_world=torch.as_tensor(obs['footprint_world'],device='cuda',dtype=torch.float32)[None,None],
                    frame_poses=torch.as_tensor(pose,device='cuda',dtype=torch.float32)[None,None])
    start=time.perf_counter()
    for j,((index,identity,rec),pub) in enumerate(zip(meta,public)):
        dest=a.output/'rows'/f'{index:06d}.json'
        if dest.exists():continue
        pvt,labels=cache[index];base,private=_collate([(pvt,labels)],'cuda');has_anchor=bool(pub['landmarks'])
        pose=rec.start_pose.copy();zero=pose.copy();zero[3]=0.;high=zero.copy();high[2]=GROUND_LEVEL[rec.map_name]+80
        planned=base['frame_poses'][0,0].cpu().numpy().copy();planned[3]=pose[3]
        variants={name:{k:base[k][:,:1] for k in ('rgb_patches','depth_patches','patch_valid','footprints_world','frame_poses')} for name in ('plan1_yaw0',)}
        variants['plan5_yaw0']={k:base[k] for k in ('rgb_patches','depth_patches','patch_valid','footprints_world','frame_poses')}
        for name,location in (('start_native',pose),('start_yaw0',zero),('start_h80_yaw0',high),('plan1_nativeyaw',planned)):
            variants[name]=frame(rec.map_name,location) if has_anchor or name.startswith('start') else variants['plan1_yaw0']
        # All inputs constructed from public instructions/map and starts. GT only below forward.
        results={};gain=float(model.visual_gain.tanh())
        for name in VARIANTS:
            kw=dict(base,**variants[name]);raw=kw['patch_valid'].clone()
            if not has_anchor:kw['patch_valid']=torch.zeros_like(raw)
            out=model(**kw);bounds=base['world_bounds'][0];target=private['target_xy'][0]*(bounds[2:]-bounds[:2])+bounds[:2]
            alternatives,diagnostic=candidate_changes(out,gain,bounds)
            result=dict(geometry=score(out.geometry_global[0],target,bounds),fused=score(out.fused_logits[0],target,bounds),
                        counterfactual={k:score(v,target,bounds) for k,v in alternatives.items()},
                        support=support_audit(out,dict(kw,_raw_valid=raw),target,bounds,has_anchor),**diagnostic)
            if name=='start_native' and not has_anchor:
                out2=model(**dict(kw,patch_valid=raw))
                result['ungated']=dict(score=score(out2.fused_logits[0],target,bounds),support=support_audit(out2,dict(kw,patch_valid=raw,_raw_valid=raw),target,bounds,False))
            results[name]=result
        atomic_json(dict(index=index,episode_id=identity,map_name=rec.map_name,instruction=rec.instruction,anchor_count=len(pub['landmarks']),
                         start_pose=pose.tolist(),planned_pose=base['frame_poses'][0,0].tolist(),variants=results),dest)
        if (j+1)%25==0:print(json.dumps(dict(completed=j+1,expected=len(meta),elapsed_s=time.perf_counter()-start)),flush=True)
    rows=[json.loads((a.output/'rows'/f'{i:06d}.json').read_text()) for i in chosen]
    summary={}
    for name in VARIANTS:
        rr=[x['variants'][name] for x in rows];g=[r['geometry']['hit'] for r in rr]
        summary[name]=dict(n=len(rr),geometry_R20=float(np.mean(g)),geometry_NE=float(np.mean([r['geometry']['NE'] for r in rr])),
                           counterfactual={k:dict(R20=float(np.mean([r['counterfactual'][k]['hit'] for r in rr])),NE=float(np.mean([r['counterfactual'][k]['NE'] for r in rr])),
                               fixes=sum(not old and r['counterfactual'][k]['hit'] for old,r in zip(g,rr)),breaks=sum(old and not r['counterfactual'][k]['hit'] for old,r in zip(g,rr))) for k in rr[0]['counterfactual']},
                           support={k:int(sum(bool(r['support'][k]) for r in rr)) for k in ('center_in_footprint','region_in_footprint','raw_depth_region_support','effective_region_support','rgb_delta_nonzero')})
    atomic_json(dict(manifest=manifest,variants=summary,elapsed_s=time.perf_counter()-start),a.output/'summary.json')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
