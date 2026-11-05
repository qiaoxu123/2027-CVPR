"""Seal TRAIN-Dev predictions before loading separate target labels."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from graphnav.checkpointing import atomic_json,atomic_save,file_sha256
from graphnav.full_relation_training import load_full_relation_model
from graphnav.complete_landmark_rgb import refine_views
from scripts.train_complete_landmark_rgb import Cache,FORMAT


def predict(a):
    torch.set_num_threads(4);a.output.mkdir(parents=True,exist_ok=True)
    qs=json.load(open(a.inventory/'queries_public.json'));ids=[q['index'] for q in qs if q['subset']=='dev']
    assert all((a.cache/'queries'/f'{i:05d}.pt').exists() for i in ids),'Every held-out query must be cached; no sampling'
    model=load_full_relation_model(torch.load(a.checkpoint,map_location='cpu',weights_only=True)).eval().requires_grad_(False);cache=Cache(a.cache)
    if a.trained:
        bundle=torch.load(a.trained,map_location='cpu',weights_only=True);assert bundle['format']==FORMAT and bundle['privileged'];model.load_state_dict(bundle['model'])
        arms=[('new_current','current',True,'real'),('new_current_zero','current',True,'zero'),('new_current_shuffle','current',True,'shuffle'),('new_whole','whole',True,'real'),('new_multi','multi',True,'real'),('new_multi_PGVB_only','multi',False,'real'),('new_multi_zero','multi',True,'zero'),('new_multi_shuffle','multi',True,'shuffle')]
    else:arms=[('geometry_relation','current',False,'static'),('old_current','current',False,'real'),('old_whole','whole',False,'real'),('old_multi','multi',False,'real')]
    for name,arm,joint,control in arms:
        target=a.output/f'{name}.pt'
        if target.exists():continue
        outputs=[];covered_counts=[];seconds=[];started=time.perf_counter()
        for start in range(0,len(ids),a.batch_size):
            base,dense,packs=cache.batch(ids[start:start+a.batch_size],arm);torch.cuda.synchronize();t=time.perf_counter()
            with torch.inference_mode():
                if control=='static':logits=base;covered=torch.zeros_like(base,dtype=torch.bool)
                else:logits,covered=refine_views(model,base,dense,packs,joint=joint,control=control)
            torch.cuda.synchronize();seconds.append(time.perf_counter()-t);outputs.append(logits.cpu());covered_counts.extend(covered.sum(-1).tolist())
            if start%400==0:print(json.dumps(dict(arm=name,queries=start,total=len(ids),seconds=time.perf_counter()-started)),flush=True)
        atomic_save(dict(ids=ids,logits=torch.cat(outputs),covered_cells=covered_counts,privileged=arm!='current',arm=name,source_SHA256=file_sha256(a.checkpoint),trained_SHA256=file_sha256(a.trained) if a.trained else None,inference_seconds=sum(seconds),wall_seconds=time.perf_counter()-started,mean_inference_ms=1000*sum(seconds)/len(ids),peak_GPU_GiB=torch.cuda.max_memory_allocated()/2**30),target)
        atomic_json(dict(path=str(target),SHA256=file_sha256(target),GT_labels_loaded=False,queries=len(ids)),a.output/f'{name}_SEALED.json')


def score(a):
    # Run only after BOTH frozen and trained predictions are immutable on disk.
    required=['geometry_relation','old_current','old_whole','old_multi','new_current','new_whole','new_multi','new_multi_PGVB_only','new_multi_zero','new_multi_shuffle','new_current_zero','new_current_shuffle']
    for name in required:assert (a.output/f'{name}_SEALED.json').exists()
    raw=json.load(open(a.data_root/'processed_citynav/citynav_train_seen.json'));objects=json.load(open(a.data_root/'cityrefer/objects.json'))
    qs=json.load(open(a.inventory/'queries_public.json'));labels=json.load(open(a.inventory/'labels_private.json'));groups=json.load(open(a.inventory/'groups.json'))
    pred={k:torch.load(a.output/f'{k}.pt',weights_only=True,map_location='cpu') for k in required};ids=pred[required[0]]['ids'];assert all(v['ids']==ids for v in pred.values())
    gt=np.array([labels[i]['target_world'] for i in ids]);bounds=np.array([groups[qs[i]['group']]['world_bounds'] if qs[i]['group'] else [0,0,1,1] for i in ids])
    # Missing reference sets retain actual map bounds; read only public prepared bounds.
    records=[json.loads(x) for x in a.train_jsonl.read_text().splitlines()];bounds=np.array([records[i]['world_bounds'] for i in ids]);size=bounds[:,2:]-bounds[:,:2]
    grid=np.stack(np.meshgrid(np.arange(28),np.arange(28)),axis=-1).reshape(-1,2);world=bounds[:,None,:2]+((grid[None]+.5)/28)*size[:,None];distance=np.linalg.norm(world-gt[:,None],axis=-1)
    per={};result={};basegood=None
    for name,v in pred.items():
        z=v['logits'];prob=z.softmax(-1).numpy();ranking=z.argsort(-1,descending=True).numpy();top=ranking[:,0];ne=distance[np.arange(len(ids)),top];good=ne<=20
        if basegood is None:basegood=good
        baseline_top=pred['geometry_relation']['logits'].argmax(-1).numpy();shift=np.linalg.norm(world[np.arange(len(ids)),top]-world[np.arange(len(ids)),baseline_top],axis=-1)
        mass=(prob*(distance<=20)).sum(-1)
        per[name]=dict(ne=ne,good=good,mass=mass,shift=shift,top=top)
        result[name]=dict(n=len(ids),R20=float(good.mean()),NE_m=float(ne.mean()),median_NE_m=float(np.median(ne)),target_20m_mass=float(mass.mean()),corrections=int((good&~basegood).sum()),damage=int((~good&basegood).sum()),net=int(good.sum()-basegood.sum()),mean_shift_m=float(shift.mean()),changed_goals=int((shift>1e-5).sum()),active_queries=sum(x>0 for x in v['covered_cells']),mean_inference_ms=v['mean_inference_ms'],wall_seconds=v['wall_seconds'],peak_GPU_GiB=v['peak_GPU_GiB'],TopK={str(k):float((np.take_along_axis(distance,ranking[:,:k],1)<=20).any(-1).mean()) for k in (1,5,15)})
    strata={}
    for i in ids:
        g=groups.get(qs[i]['group'],{});types=set(g.get('types',[]));cat='Building' if types=={'Building'} else 'Road' if types and types<={'TrafficRoad','Footpath','Bridge','Road'} else 'Mixed_or_Other'
        side=g.get('plan',{}).get('whole',{}).get('side_m',0.)
        category=objects[qs[i]['map_name']].get(str(raw[i]['object_ids'][0]),{}).get('object_type','Unknown')
        strata[i]=dict(map=qs[i]['map_name'],reference_category=cat,target_category=category,reference_extent='small_<100m' if side<100 else 'medium_100-250m' if side<250 else 'large_>=250m',group=qs[i]['group'])
    bygroup={}
    for field in ('map','reference_category','target_category','reference_extent'):
        bygroup[field]={}
        for value in sorted(set(strata[i][field] for i in ids)):
            mask=np.array([strata[i][field]==value for i in ids]);bygroup[field][value]={name:dict(n=int(mask.sum()),R20=float(d['good'][mask].mean()),NE_m=float(d['ne'][mask].mean()),corrections=int((d['good'][mask]&~basegood[mask]).sum()),damage=int((~d['good'][mask]&basegood[mask]).sum())) for name,d in per.items()}
    rng=np.random.default_rng(42);maps=sorted(set(strata[i]['map'] for i in ids));map_indices={m:np.flatnonzero([strata[i]['map']==m for i in ids]) for m in maps};ci={}
    for name in required[1:]:
        diffs=[]
        for _ in range(2000):
            drawn=rng.choice(maps,len(maps),replace=True);ix=np.concatenate([map_indices[m] for m in drawn]);diffs.append(float((per[name]['good'][ix].astype(float)-basegood[ix]).mean()))
        ci[name]=dict(map_group_bootstrap_delta_R20_95CI=np.quantile(diffs,[.025,.975]).tolist(),maps=len(maps))
    rows=[]
    for j,i in enumerate(ids):rows.append(dict(index=i,**strata[i],methods={k:{'NE_m':float(v['ne'][j]),'R20':bool(v['good'][j]),'mass20':float(v['mass'][j]),'shift_m':float(v['shift'][j])} for k,v in per.items()}))
    atomic_json(dict(methods=result,strata=bygroup,CI=ci,protocol='position proxy, not pixel instance accuracy; frozen core previously trained on all TRAIN; no Val/Test'),a.output/'metrics.json');atomic_json(rows,a.output/'paired.json');print(json.dumps(result,indent=2))


def main():
    p=argparse.ArgumentParser()
    for k in ('inventory','cache','checkpoint','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--data-root',type=Path);p.add_argument('--trained',type=Path);p.add_argument('--batch-size',type=int,default=4);p.add_argument('--score',action='store_true');p.add_argument('--train-jsonl',type=Path);a=p.parse_args()
    if a.score:score(a)
    else:predict(a)

if __name__=='__main__':main()
