"""Frozen Val-Seen ORACLE offline diagnosis. Never used by deployable inference."""
import argparse,json,hashlib,csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from shapely.geometry import Polygon
from scripts.oracle_bearing_geometry import rerank,WEIGHTS
from graphnav.checkpointing import file_sha256


def key(map_name,instruction,pose):
    return hashlib.sha256(json.dumps([map_name,instruction,list(pose)],sort_keys=True).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    for n in ['baseline','annotations','public-objects','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    raw=json.loads(a.annotations.read_text());public=json.loads(a.public_objects.read_text())
    rows=list(map(json.loads,(a.baseline/'predictions.jsonl').read_text().splitlines()));assert len(raw)==len(rows)==2470
    angles={};records=[];summary=defaultdict(lambda:defaultdict(list));storage={};seen={}
    for row,task in zip(rows,raw):
        i=row['index'];assert task['object_ids'][0]==row['episode_id'][1] and task['ann_ids'][0]==row['episode_id'][2]
        folder=a.baseline/f'belief_steps/{i:05d}';rec=json.loads((folder/'public_record.json').read_text());obs=np.load(folder/'00.npz')
        bounds=np.asarray(rec['world_bounds']);span=bounds[2:]-bounds[:2];xy=obs['grid']*span+bounds[:2];z=torch.from_numpy(obs['full']).float();gt=np.asarray(task['target_positions'][-1])[:2]
        refs=rec['landmarks'];anchors=[];valid=True
        for r in refs:
            poly=np.asarray(r['contour'])*span+bounds[:2]
            if len(poly)<3 or not Polygon(poly).is_valid or Polygon(poly).area<=0:valid=False;break
            anchors.append(poly.mean(0))
        anchors=np.asarray(anchors).reshape(-1,2)
        names=[r['name'].lower().strip() for r in refs];same=len(names)!=len(set(names))
        global_ambiguous=any(sum(str(o.get('name','')).lower().strip()==n for o in public[rec['map_name']].values())>1 for n in names)
        group='no_anchor' if not refs else 'invalid_polygon' if not valid else 'same_name_multi_polygon' if same or global_ambiguous else 'single_unique_polygon' if len(refs)==1 else 'multiple_distinct_names'
        t=[]
        if valid and len(anchors):
            delta=gt[None,:]-anchors
            if (np.linalg.norm(delta,axis=1)<1e-9).any():group='undefined_angle_abstain'
            else:t=np.arctan2(delta[:,1],delta[:,0]).tolist()
        # PRIVATE coordinate discarded before runtime artifact creation. All
        # fixed bound anchors retained, including same-name multiple polygons.
        k=key(rec['map_name'],rec['instruction'],row['poses'][0]);entry=dict(angles=t,anchors=anchors.tolist() if t else [],group=group,landmark_ids=[r['id'] for r in refs],world_bounds=bounds.tolist(),public_binding_SHA256=hashlib.sha256(json.dumps(refs,sort_keys=True).encode()).hexdigest())
        if k in angles and angles[k]!=entry:raise ValueError('Conflicting duplicated public task key; never pick GT')
        angles[k]=entry
        dist=np.linalg.norm(xy-gt,axis=1);nearest=int(dist.argmin());cell=float(np.linalg.norm(span/28));result=dict(index=i,episode_id=row['episode_id'],group=group,N_anchors=len(refs),grid_min_NE=float(dist.min()),grid_support20=bool(dist.min()<=20),GT=gt.tolist(),instruction=rec['instruction']);scores={}
        for name in ['A','B','C',*[f'D{w:g}' for w in WEIGHTS]]:
            q=rerank(z,xy,anchors if t else [],t,name[0],float(name[1:]) if name.startswith('D') else 1.,cell)
            order=q.argsort(descending=True,stable=True).numpy();hit=np.flatnonzero(dist[order]<=20)
            m=dict(NE=float(dist[order[0]]),R20=bool(dist[order[0]]<=20),R5=bool(dist[order[:5]].min()<=20),R10=bool(dist[order[:10]].min()<=20),rank_nearest=int(np.flatnonzero(order==nearest)[0]+1),rank_first20=int(hit[0]+1) if len(hit) else None,top1=int(order[0]),mass20=float(q.softmax(-1)[torch.from_numpy(dist<=20)].sum()))
            result[name]=m;scores[name]=q.numpy()
            for g in ['all',group]:summary[g][name].append(m)
        result['A_top20_support']=bool(dist[z.topk(20).indices.numpy()].min()<=20)
        result['A_selected_grid_error']=float(dist[result['A']['top1']]);records.append(result)
        storage[str(i)]=dict(grid=xy,scores=scores,anchors=anchors,angles=t,polygons=[(np.asarray(r['contour'])*span+bounds[:2]).tolist() for r in refs],bounds=bounds,target=gt)
    metrics={}
    for group,arms in summary.items():
        metrics[group]={}
        for arm,ms in arms.items():
            baseline=arms['A'];metrics[group][arm]={k:float(np.mean([v[k] for v in ms])) for k in ['NE','R20','R5','R10','rank_nearest','mass20']}
            metrics[group][arm].update(N=len(ms),fixes=sum(not b['R20'] and m['R20'] for b,m in zip(baseline,ms)),damage=sum(b['R20'] and not m['R20'] for b,m in zip(baseline,ms)))
    (a.output/'angles_ONLY_ORACLE.json').write_text(json.dumps(dict(ORACLE=True,GT_range_supplied=False,tasks=angles),indent=2))
    (a.output/'offline_rows.json').write_text(json.dumps(records,indent=2));(a.output/'offline_summary.json').write_text(json.dumps(metrics,indent=2));torch.save(storage,a.output/'offline_fields.pt')
    manifest=dict(privileged_diagnostic=True,closedloop_arms=['B','C','D1'],weights=WEIGHTS,baseline_SHA=file_sha256(a.baseline/'metrics.json'),source_SHA=file_sha256(a.annotations),angle_file_SHA=file_sha256(a.output/'angles_ONLY_ORACLE.json'),grid_support_missing=sum(not r['grid_support20'] for r in records),baseline_top20_missing=sum(not r['A_top20_support'] for r in records),no_GT_target_or_range_at_runtime=True)
    (a.output/'FROZEN_DESIGN.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(metrics,indent=2),flush=True)
if __name__=='__main__':main()
