"""Observed public cache and separate private target-derived supervision."""
from common import *
import argparse
import time
import cv2
import numpy as np
import torch
from geometry import sample_surface
from labels import associate
from schema import SCHEMA_VERSION
from policy import RefinementPolicy
from graphnav.landmark_projection import world_to_frame
from graphnav.closedloop import PublicEpisode,PublicCatalog
from graphnav.dynamic_runtime import NativeObservationBuilder
from sbfnav.dataset import CityNavDataset,CityReferCatalog
from sbfnav.planning_state import PlanningStateBuilder
from sbfnav.mapdata import GROUND_LEVEL


class DetailBuilder(NativeObservationBuilder):
    def build_from_poses(self,episode,poses):
        state=super().build_from_poses(episode,poses);footprint=state.current_observation['footprint_world']
        px=np.array([[self.renderer._raster.index(float(x),float(y))[1],self.renderer._raster.index(float(x),float(y))[0]] for x,y in footprint])
        span=np.linalg.norm(px-np.roll(px,-1,axis=0),axis=-1).max();n=int(min(4096,max(384,32*np.ceil((span+1)/32))))
        try:
            self.renderer.image_size=n;state.detail_observation=self.renderer.current_rgbd(episode.map_name,poses[-1],GROUND_LEVEL[episode.map_name])
            state.detail_observation.pop('surface_height_m',None)
        finally:self.renderer.image_size=384
        return state


def private_labels(record,catalog,public,meta,ob):
    target=np.asarray(record.target_position[:2],np.float32);candidates=meta['candidates'];refs=meta['references'];n=len(candidates)
    objects=[]
    if n:
        allpolys=np.concatenate([r['poly'] for r in candidates]);lo=allpolys.min(0);hi=allpolys.max(0)
        for oid,obj in catalog.objects[record.map_name].items():
            kind={'Car':0,'Building':1}.get(obj['object_type'],-1)
            if kind<0:continue
            p=np.asarray(obj['contour'],np.float32)
            if (p.max(0)<lo).any() or (p.min(0)>hi).any():continue
            objects.append((oid,kind,p))
    a=associate([r['poly'] for r in candidates],[r['kind'] for r in candidates],objects,record.object_id)
    xy=np.asarray([r['xy'] for r in candidates],np.float32).reshape(-1,2);rxy=np.asarray([r['xy'] for r in refs],np.float32).reshape(-1,2)
    delta=target[None]-rxy;distance=np.linalg.norm(delta,axis=-1);angle=delta/np.maximum(distance[:,None],1e-6)
    obj=catalog.target_object(record);p=world_to_frame(np.asarray(obj['contour']),ob['footprint_world'],ob['bgr'].shape)
    valid=ob['depth_valid']&(ob['bgr'].max(-1)>0);surface=ob['pose'][2]-ob['depth_m']
    stat=sample_surface(surface,valid,p);known=np.array([r['surface']['valid'] and stat['valid'] for r in refs],bool)
    dh=np.array([(stat['z']-r['surface']['z'])/20 if k else 0 for r,k in zip(refs,known)],np.float32)
    offset=target[None]-xy;limit=public['network']['offset_limit'].cpu().numpy()
    target_pixel=world_to_frame(target[None],ob['footprint_world'],ob['bgr'].shape)[0];h,w=ob['bgr'].shape[:2]
    visible=bool(np.isfinite(target_pixel).all() and 0<=target_pixel[0]<w and 0<=target_pixel[1]<h)
    if visible:visible=bool(ob['bgr'][int(target_pixel[1]),int(target_pixel[0])].max()>0)
    ov=(a['ranking']>0)&(np.abs(offset)<=limit).all(-1)&visible
    # Positive labels require a reachable local correction. All known nonmatches
    # can label no-match; ambiguous/unknown identities never label absence.
    match=bool(ov.any());match_valid=match or n==0 or (bool(a['rank_label_mask'].all()) and not bool((a['ranking']>0).any()))
    labels=dict(ranking=a['ranking'],rank_label_mask=a['rank_label_mask'],angle=angle,angle_valid=distance>1,
        distance=np.log1p(distance)/np.log1p(configuration()['radius']),height=dh,height_valid=known,
        offset=offset,offset_valid=ov,match=float(match),match_valid=match_valid,target_xy=target)
    return {k:torch.as_tensor(v) for k,v in labels.items()},dict(association=a['association'],target_surface=stat,target_visible=visible,
        target_polygon=np.asarray(obj['contour']).tolist(),target_kind={'Car':0,'Building':1}.get(obj['object_type'],-1))


def cache_signature():
    names=['prepare.py','perception.py','geometry.py','labels.py','schema.py','policy.py','common.py']
    return dict(schema_version=SCHEMA_VERSION,config_sha256=sha(ROOT/'configs/frozen.json'),extractor={n:sha(ROOT/'scripts'/n) for n in names})


def causal_extract(steps,cached,compute):
    missing=[step for step in steps if step not in cached]
    last=max(missing) if missing else -1
    for step in steps:
        value=compute(step) if step<=last else None
        yield step,cached[step] if step in cached else value


def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int);args=p.parse_args()
    torch.set_num_threads(2);cv2.setNumThreads(1);c=configuration();verify_sources(c);signature=cache_signature()
    sigpath=ROOT/'data/signature.json'
    if sigpath.exists() and json.loads(sigpath.read_text())!=signature:raise ValueError('Cache source signature changed; use a new cache version')
    save(signature,sigpath);ds=CityNavDataset(DATA,'train_seen',version=c['data_version']);cat=CityReferCatalog(DATA/'cityrefer')
    builder=DetailBuilder(PlanningStateBuilder(DATA,PublicCatalog(cat),canvas_size=384,field_size=28),RGBD,ROOT/'runs/render_unused')
    policy=RefinementPolicy(ROOT/'runs/cache_query',collect=True);entries=[];started=time.perf_counter()
    for split in ['train','calibration','dev']:
        chosen=c['splits'][split][:args.limit] if args.limit else c['splits'][split]
        for count,entry in enumerate(chosen):
            rec=ds[entry['index']];g=GROUND_LEVEL[rec.map_name];trajectory=np.asarray(rec.trajectory)
            legal=np.flatnonzero((trajectory[:,2]-g>=10)&(trajectory[:,2]-g<=100)&np.isfinite(trajectory).all(-1))
            if not len(legal):continue
            steps=sorted({int(legal[round(q*(len(legal)-1))]) for q in c['quantiles']})
            public_episode=PublicEpisode(rec.map_name,rec.instruction,rec.start_pose.copy(),tuple(cat.referenced_landmarks(rec)),tuple(cat.referenced_landmark_names(rec)))
            policy.set_episode(public_episode,builder.transform(rec.map_name),cat)
            cached={}
            for step in steps:
                path=ROOT/f'data/{split}_{entry["index"]}_{step}.pt'
                if path.exists():
                    cached[step]=torch.load(path,map_location='cpu',weights_only=False)
                    if cached[step]['signature']!=signature:raise ValueError('Existing row signature mismatch')
            def compute(step):
                state=builder.build_from_poses(public_episode,trajectory[:step+1]);policy.predict(state,rec.instruction)
                public={k:{name:v.detach().cpu().clone() for name,v in value.items()} for k,value in policy.pack.items()}
                labels,private=private_labels(rec,cat,public,policy.meta,state.detail_observation)
                return dict(public=public,labels=labels,meta=policy.meta,private=private,episode=entry['index'],step=step,
                    target_instance=[rec.map_name,int(rec.object_id)],instruction=rec.instruction,signature=signature)
            for step,row in causal_extract(steps,cached,compute):
                path=ROOT/f'data/{split}_{entry["index"]}_{step}.pt'
                if step not in cached:
                    tmp=path.with_suffix('.tmp');torch.save(row,tmp);tmp.replace(path)
                entries.append(dict(split=split,path=str(path),episode=entry['index'],step=step,content_sha256=sha(path)))
            save(dict(status='extracting',split=split,episodes=count+1,total=len(chosen),views=len(entries),seconds=time.perf_counter()-started),ROOT/'reports/preparation_status.json')
            print(f'{split} {count+1}/{len(chosen)} views={len(entries)}',flush=True)
    save(dict(complete=not bool(args.limit),signature=signature,entries=entries),ROOT/'data/manifest.json')
    save(dict(status='complete' if not args.limit else 'partial_smoke',views=len(entries),seconds=time.perf_counter()-started),ROOT/'reports/preparation_status.json')


if __name__=='__main__':main()
