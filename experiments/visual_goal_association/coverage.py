"""P1 visited-camera coverage; planner public-only, GT visibility evaluator-only."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from shapely.geometry import Polygon
from graphnav.renderer import CityNavObservedViewRenderer
from graphnav.memory import ObservedRGBMemory
from graphnav.active_observation import plan_active_observation,ActiveObservationConfig
from graphnav.landmark_projection import project_landmark,world_to_frame
from graphnav.semantic_geometry import valid_surface_polygon
from experiments.bidirectional_core.cache import CachedCoreDataset,jsonl,DEFAULT_DATA,public_landmarks
from experiments.bidirectional_core.vision import select_regions
from experiments.bidirectional_core.stats import summarize_rows,paired_difference
from graphnav.checkpointing import file_sha256
from .evaluate import write_json,write_rows,SEED


def frame_geometry(observation,landmarks):
    """Same ROI rules as frozen cache, plus actual depth-support diagnostics."""
    regions=select_regions(observation,landmarks,max_rois=32)
    quality={}
    byid={l['id']:l for l in landmarks.candidates(np.min(observation['footprint_world'],0),np.max(observation['footprint_world'],0))}
    for r in regions:
        roi=project_landmark(byid[r['object_id']],observation['footprint_world'],observation['bgr'].shape,min_visible=.9,min_pixels=256)
        depth=float(observation['depth_valid'][roi.mask.astype(bool)].mean())
        quality[r['object_id']]=dict(native_min_side=r['native_min_side'],depth_support=depth,credible_geometry=depth>=.9,quality=float(r['quality'][1]*r['quality'][2]),grid_center=r['center_world'])
    return regions,quality


def diagnostic_target(observation,target,selected,quality):
    """Offline reason tree. Cannot be called by the observation planner."""
    if target is None:return dict(reason='missing_target_annotation',fov=False,valid=False,credible=False)
    contour=np.asarray(target.get('contour',[]),float)
    if not valid_surface_polygon(contour):return dict(reason='invalid_polygon',fov=None,valid=False,credible=False)
    view=Polygon(observation['footprint_world']);poly=Polygon(contour)
    if not poly.intersects(view) or poly.intersection(view).area<=1e-8:
        return dict(reason='outside_fov',fov=False,valid=False,credible=False)
    roi=project_landmark(dict(id=target['id'],contour=contour),observation['footprint_world'],observation['bgr'].shape,min_visible=0,min_pixels=1)
    if roi is None:return dict(reason='projection_or_subpixel_failure',fov=True,valid=False,credible=False)
    mask=roi.mask.astype(bool);rgb=float((observation['bgr'].max(-1)>0)[mask].mean());depth=float(observation['depth_valid'][mask].mean())
    side=min(roi.bbox_xyxy[2]-roi.bbox_xyxy[0],roi.bbox_xyxy[3]-roi.bbox_xyxy[1])
    oid=target['id'];valid=oid in selected
    if roi.visible_fraction<.9:reason='fov_clipped'
    elif roi.projected_area_px<256 or roi.visible_pixels<256 or side<8:reason='too_few_native_pixels'
    elif rgb<.9:reason='insufficient_rgb_source'
    elif not valid:
        # Distinguish patch support from the existing source-independent ROI cap.
        uncapped,_=frame_geometry_uncapped(observation,target)
        reason='roi_quality_cap_32' if uncapped else 'patch_support_or_projection_registration'
    elif depth<.9:reason='valid_roi_depth_insufficient'
    else:reason='valid_geometric_roi_semantics_unverified'
    return dict(reason=reason,fov=True,valid=valid,credible=valid and depth>=.9,depth_support=depth,rgb_support=rgb,native_min_side=side,visible_fraction=roi.visible_fraction,projected_area=roi.projected_area_px)


def frame_geometry_uncapped(observation,target):
    index=public_landmarks({'map':{target['id']:target}})['map']
    rows=select_regions(observation,index,32)
    return bool(rows),None


def public_move(pose,bounds,ground,grid_xy,logits,seen,instruction,strategy,used):
    """No target/annotation argument. Bounded extra observation only."""
    pose=np.asarray(pose,float);bounds=np.asarray(bounds,float);height=pose[2]-ground
    legal=10<=height<=100 and (pose[:2]>=bounds[:2]).all() and (pose[:2]<=bounds[2:]).all()
    if not legal:return None
    if strategy=='hold':return dict(action='hold',pose=pose.copy(),candidate=None)
    best=int(np.argmax(logits));goal=bounds[:2]+np.asarray(grid_xy[best])*(bounds[2:]-bounds[:2])
    nextpose=pose.copy();action=strategy
    if strategy=='approach':
        delta=goal-pose[:2];nextpose[:2]+=delta*min(1,50/max(np.linalg.norm(delta),1e-12))
    elif strategy=='descend':nextpose[2]=max(ground+10,pose[2]-10)
    elif strategy=='lateral':
        delta=goal-pose[:2]
        if np.linalg.norm(delta)<1e-8:delta=np.array([np.cos(pose[3]),np.sin(pose[3])])
        delta/=np.linalg.norm(delta);nextpose[:2]+=((-1 if used%2 else 1)*12*np.array([-delta[1],delta[0]]))
    elif strategy=='belief_active':
        decision=plan_active_observation(pose_xyzyaw=pose,map_bounds=bounds,ground_level=ground,candidate_xy=grid_xy,candidate_logits=logits,observed_confidence=seen,instruction=instruction,step=used,used_actions=used,config=ActiveObservationConfig(max_actions=2,horizon=20))
        if decision is None:return dict(action='hold_resolved',pose=pose.copy(),candidate=None)
        nextpose[:3]=decision.xyz;action=decision.action;best=decision.candidate_id
    else:raise ValueError('invalid public policy')
    nextpose[:2]=np.clip(nextpose[:2],bounds[:2],bounds[2:])
    assert np.linalg.norm(nextpose[:2]-pose[:2])<=50+1e-6 and abs(nextpose[2]-pose[2])<=10+1e-6
    assert 10-1e-6<=nextpose[2]-ground<=100+1e-6
    return dict(action=action,pose=nextpose,candidate=best)


def stats(rows):
    result={k:summarize_rows(rows,k,repetitions=2000,seed=SEED) for k in rows[0]['metrics']}
    eligible=[r for r in rows if r['metrics']['error_credible']]
    result['independent_error_credible']=dict(states=len(eligible),families=len({r['family_id'] for r in eligible}),objects=len({(r['map_name'],r['target_object_id']) for r in eligible}))
    return result


def run(cache,output,all_history=True,data_root=DEFAULT_DATA):
    root=Path(output)
    if root.exists():raise ValueError('no overwrite')
    root.mkdir(parents=True);started=time.perf_counter();ds=CachedCoreDataset(cache)
    raw=json.loads((Path(data_root)/'cityrefer/objects.json').read_text());indexes=public_landmarks(raw)
    episodepath=[p for p in ds.manifest['frozen_resources'] if Path(p).name=='pilot_episodes.jsonl'][0]
    assert file_sha256(episodepath)==ds.manifest['frozen_resources'][episodepath]
    eps={r['episode_id']:r for r in jsonl(episodepath)}
    renderer=CityNavObservedViewRenderer(Path(data_root)/'rgbd',root/'environment_private',image_size=384)
    policies=['current','cached_history','dense_causal_history','hold','approach','descend','lateral','belief_active']
    results={s:[] for s in policies};history={};cached={};indices=sorted([i for i in range(len(ds)) if ds[i]['metadata']['direction']=='forward'],key=lambda i:(ds[i]['metadata']['map_name'],ds[i]['metadata']['episode_id'],ds[i]['metadata']['cutoff_step']))
    source_differences=[];observations=0;frame_seconds=0;reasons={};cases=0
    for completed,i in enumerate(indices):
        r=ds[i];p=r['public'];m=r['metadata'];pose=p['pose_world'].numpy();ground=float(p['ground_level']);episode=eps[m['episode_id']]
        h=history.setdefault(m['episode_id'],dict(step=-1,quality={},frames=0,seconds=0));ch=cached.setdefault(m['episode_id'],{})
        oid=str(episode['private_labels']['target_object_id']);target=dict(raw[m['map_name']][oid],id=oid) if oid in raw[m['map_name']] else None
        def observe(v):
            nonlocal observations,frame_seconds
            t=time.perf_counter()
            if v[2]-ground<=0:return None,[],{},time.perf_counter()-t
            obs=renderer.current_rgbd(m['map_name'],v,ground);regs,q=frame_geometry(obs,indexes[m['map_name']]);seconds=time.perf_counter()-t;observations+=1;frame_seconds+=seconds
            return obs,regs,q,seconds
        # Replay actual visited human poses only through this prefix cutoff.
        current=None
        for step in range(h['step']+1,m['cutoff_step']+1):
            v=np.asarray(episode['trajectory_xyzyaw'][step]);obs,regs,q,secs=observe(v)

            for key,val in q.items():
                if key not in h['quality'] or val['credible_geometry']:h['quality'][key]=val
            h['frames']+=int(obs is not None);h['seconds']+=secs
            if step==m['cutoff_step']:current=(obs,regs,q,secs)
        h['step']=m['cutoff_step']
        if current is None:current=observe(pose)
        obs,regs,q,secs=current
        # Verify unchanged cached current ROI selection, not silently repairing.
        ids=[v['object_id'] for v in regs];cacheids=[v['object_id'] for j,v in enumerate(m['roi_objects']) if p['roi_valid'][j]]
        if ids!=cacheids:source_differences.append(dict(sample_id=m['sample_id'],rendered=ids,cached=cacheids))
        for key,val in q.items():
            if key not in ch or val['credible_geometry']:ch[key]=val
        if obs is not None:diag=diagnostic_target(obs,target,ids,q)
        else:diag=dict(reason='invalid_agl',fov=False,valid=False,credible=False)
        reasons[diag['reason']]=reasons.get(diag['reason'],0)+1
        xy=p['bounds'][:2].numpy()+p['grid_xy'].numpy()*(p['bounds'][2:]-p['bounds'][:2]).numpy()
        distance=np.linalg.norm(xy-r['labels']['target_world'][:2].numpy(),axis=1);static=int(p['static_logits'].argmax());wrong=bool(distance[static]>20);top8=np.argsort(-p['static_logits'].numpy())[:8]
        def coverage(qdict):
            cells=set()
            for val in qdict.values():cells.add(int(np.square(xy-val['grid_center']).sum(1).argmin()))
            return len(cells)/784,sum(int(k in cells) for k in top8)/8
        denseq=h['quality']
        for policy in policies:
            observed=dict(q);actions=[];extra_seconds=0;travel=0;last=pose.copy();trajectory=[]
            if policy=='cached_history':observed=dict(ch)
            elif policy=='dense_causal_history':observed=dict(denseq)
            elif policy not in ('current',):
                seen=np.zeros(784)
                for val in observed.values():seen[int(np.square(xy-val['grid_center']).sum(1).argmin())]=val['quality']
                for action_step in range(2):
                    move=public_move(last,p['bounds'].numpy(),ground,p['grid_xy'].numpy(),p['static_logits'].numpy(),seen,m['instruction'],policy,action_step)
                    if move is None:break
                    v=move['pose'];newobs,newregs,newq,cost=observe(v);extra_seconds+=cost;travel+=float(np.linalg.norm(v[:3]-last[:3]));actions.append(move['action']);trajectory.append(v.tolist());last=v
                    for key,val in newq.items():
                        if key not in observed or val['credible_geometry']:observed[key]=val
                        seen[int(np.square(xy-val['grid_center']).sum(1).argmin())]=max(seen[int(np.square(xy-val['grid_center']).sum(1).argmin())],val['quality'])
                    # Policy receives only frames it actually visited. Save evidence
                    # of a fixed representative subset, selected independently of GT.
                    if newobs is not None and completed%32==0:
                        dest=root/'observations'/m['sample_id'].replace(':','_')/policy
                        dest.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(dest/f'{action_step}.png'),newobs['bgr'])
            valid=oid in observed;credible=valid and observed[oid]['credible_geometry'];cells,top=coverage(observed)
            metrics=dict(target_valid=float(valid),target_credible=float(credible),static_error=float(wrong),error_visible=float(wrong and valid),error_credible=float(wrong and credible),candidate_cell_coverage=cells,top8_candidate_coverage=top,extra_actions=len(actions),extra_travel_m=travel,added_observation_seconds=extra_seconds,causal_history_frames=h['frames'] if policy=='dense_causal_history' else 0,history_replay_seconds=h['seconds'] if policy=='dense_causal_history' else 0)
            results[policy].append(dict(sample_id=m['sample_id'],family_id=m['family_id'],map_name=m['map_name'],cutoff_step=m['cutoff_step'],target_object_id=oid,metrics=metrics,current_diagnostic=diag,added_actions=actions,visited_poses=trajectory,observed_object_ids=sorted(observed)))
        if completed%32==0:print(json.dumps(dict(completed=completed+1,total=len(indices),observations=observations,seconds=time.perf_counter()-started)),flush=True)
    for policy,rows in results.items():write_rows(root/(policy+'_predictions.jsonl'),rows)
    summaries={s:stats(r) for s,r in results.items()}
    paired={s:{k:paired_difference(results['current'],r,k,repetitions=2000,seed=SEED) for k in ['target_valid','target_credible','error_credible','top8_candidate_coverage']} for s,r in results.items()}
    activepaired={s:paired_difference(results['hold'],results[s],'target_credible',repetitions=2000,seed=SEED) for s in ['approach','descend','lateral','belief_active']}
    write_json(root/'metrics.json',dict(models=summaries,paired_vs_current=paired,paired_vs_same_budget_hold=activepaired,active_gates={s:v['map_cluster_95ci'][0]>0 for s,v in activepaired.items()},current_failure_reasons=reasons,cache_selection_differences=source_differences,actual_rendered_views=observations,render_projection_seconds=frame_seconds,total_seconds=time.perf_counter()-started,seed=SEED,cache_manifest_sha256=file_sha256(Path(cache)/'manifest.json'),protocol_sha256=file_sha256(Path(__file__).with_name('protocol.json')),note='environment-generated visited virtual observations; no semantic encoding or navigation rollout; depth is vertical proxy; all geometric ROIs weak, human verified attributes 0'))
    print(json.dumps(dict(source_differences=len(source_differences),reasons=reasons,models={s:{k:v[k]['mean'] for k in ['target_valid','target_credible','extra_actions']} for s,v in summaries.items()})),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.cache,a.output)
if __name__=='__main__':main()
