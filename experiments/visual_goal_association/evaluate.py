"""Frozen P0 association experiment. All target access isolated to evaluator."""
import argparse
import itertools
import json
from pathlib import Path
import time
import numpy as np
import torch
from graphnav.checkpointing import file_sha256
from graphnav.memory import ObservedRGBMemory
from experiments.bidirectional_core.cache import CachedCoreDataset,jsonl,DEFAULT_DATA
from experiments.bidirectional_core.stats import summarize_rows,paired_difference
from experiments.rgb_grounding.evaluate_features import load
from experiments.rgb_grounding.goal_evidence import load_head,encode_scores
from .association import METHODS,associate,calibrated_scores,fuse,identity_oracle

SEED=20261011


def write_json(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def write_rows(path,rows):Path(path).write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))


def prepare(cache,features,controls,checkpoint,output,data_root=DEFAULT_DATA):
    output=Path(output)
    if output.exists():raise ValueError('no overwrite')
    output.mkdir(parents=True);started=time.perf_counter()
    dataset=CachedCoreDataset(cache);source=load(features);control={r['sample_id']:r for r in load(controls)}
    assert set(control)=={r['metadata']['sample_id'] for r in source}
    head=load_head(checkpoint,'cpu');head.requires_grad_(False)
    records={kind:encode_scores(source,head,control,kind,'cpu') for kind in ['real','zero','shuffled']}
    objects=json.loads((Path(data_root)/'cityrefer/objects.json').read_text())
    episodes_path=[p for p in dataset.manifest['frozen_resources'] if Path(p).name=='pilot_episodes.jsonl'][0]
    assert file_sha256(episodes_path)==dataset.manifest['frozen_resources'][episodes_path]
    episodes={r['episode_id']:r for r in jsonl(episodes_path)}
    public=[];labels={};association_cache={};diagnostic=[]
    for i in range(len(dataset)):
        r=dataset[i];p=r['public'];m=r['metadata']
        if m['direction']!='forward':continue
        item={k:m[k] for k in ['sample_id','episode_id','family_id','map_name','cutoff_step','instruction']}
        item.update(logits=p['static_logits'].numpy(),grid_xy=p['grid_xy'].numpy(),bounds=p['bounds'].numpy(),pose=p['pose_world'].numpy(),ground=float(p['ground_level']),regions=[])
        pose=item['pose'];height=pose[2]-item['ground']
        footprint=ObservedRGBMemory.view_corners(pose[:2],pose[3],height) if height>0 else None
        current=records['real'].get(m['sample_id'])
        if current:
            s=current['row']
            for k in ['sample_id','episode_id','map_name','cutoff_step','actual_camera_pose']:
                assert s['metadata'][k]==m[k],('source binding',k)
            assert [a['object_id'] for a in s['metadata']['roi_objects']]==[a['object_id'] for a in m['roi_objects']]
            assert torch.equal(s['public']['roi_valid'],p['roi_valid'])
        allregions=[]
        for j,roi in enumerate(m['roi_objects']):
            if not p['roi_valid'][j]:continue
            oid=roi['object_id'];contour=objects[m['map_name']][oid]['contour']
            key=(m['sample_id'],oid)
            weights={method:associate(contour,footprint,item['bounds'],item['grid_xy'],method,int(p['roi_grid_ids'][j])) for method in METHODS}
            association_cache[key]=weights
            reg=dict(object_id=oid,object_type=roi['object_type'],quality=float(p['roi_quality'][j,1]*p['roi_quality'][j,2]),native_min_side=roi['native_min_side'],step=m['cutoff_step'],weights=weights,grid=int(p['roi_grid_ids'][j]))
            allregions.append(reg)
            if current:
                reg=dict(reg,scores={kind:float(records[kind][m['sample_id']]['score'][0,j]) for kind in records},confidences={kind:float(records[kind][m['sample_id']]['confidence'][0,j]) for kind in records})
                item['regions'].append(reg)
        item['all_regions']=allregions
        public.append(item)
        # Private information is sealed in evaluation labels, never in public.
        ep=episodes[m['episode_id']]
        labels[m['sample_id']]=dict(target_world=r['labels']['target_world'][:2].numpy(),target_object_id=str(ep['private_labels']['target_object_id']),weak_eligible=bool(r['labels']['positive_roi_mask'].any() and (r['labels']['semantic_valid_mask'] & ~r['labels']['positive_roi_mask']).any()))
    public.sort(key=lambda r:(r['episode_id'],r['cutoff_step']))
    torch.save(dict(public=public,labels=labels),output/'prepared.pt')
    provenance=dict(complete=True,states=len(public),feature_states=sum(bool(r['regions']) for r in public),cached_states=len(dataset),
        checkpoint_sha256=file_sha256(checkpoint),inputs={str(Path(q).resolve()):file_sha256(Path(q)/'manifest.json') for q in [cache,features,controls]},episodes_sha256=file_sha256(episodes_path),objects_sha256=file_sha256(Path(data_root)/'cityrefer/objects.json'),prepared_sha256=file_sha256(output/'prepared.pt'),seconds=time.perf_counter()-started,seed=SEED,no_backbone_encoding=True)
    write_json(output/'manifest.json',provenance);print(json.dumps(provenance),flush=True)


def load_prepared(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['complete'] and file_sha256(root/'prepared.pt')==manifest['prepared_sha256']
    # NumPy numerical arrays only; this experiment owns these bound local files.
    with torch.serialization.safe_globals([np.core.multiarray._reconstruct,np.ndarray,np.dtype,type(np.dtype('float64')),type(np.dtype('float32'))]):
        return torch.load(root/'prepared.pt',map_location='cpu',weights_only=True)


def observed_candidates(public,history,kind,all_rois=False):
    memory={}
    for state in public:
        bank=memory.setdefault(state['episode_id'],{})
        current=state['all_regions'] if all_rois else state['regions']
        for reg in current:
            # One identity per episode, quality-weighted actual observations;
            # weights union only footprints already visited, never whole map.
            prev=bank.get(reg['object_id'])
            if prev is None:bank[reg['object_id']]=[reg]
            else:prev.append(reg)
        groups=list(bank.values()) if history else [[r] for r in current]
        merged=[]
        for group in groups:
            assert all(r['step']<=state['cutoff_step'] for r in group)
            q=np.array([r['quality'] for r in group]);total=max(q.sum(),1e-12)
            entry=dict(object_id=group[0]['object_id'],object_type=group[0]['object_type'],quality=float(q.max()),views=len(group),weights={m:np.max([r['weights'][m] for r in group],axis=0) for m in METHODS})
            if not all_rois:
                entry.update(score=float(sum(r['scores'][kind]*qq for r,qq in zip(group,q))/total),confidence=float(sum(r['confidences'][kind]*qq for r,qq in zip(group,q))/total))
            merged.append(entry)
        yield state,merged


def public_prediction(state,regions,config,kind):
    """Normal prediction receives no target identity, labels or coordinates."""
    logits=state['logits'];method=config['mapping']
    weights=np.stack([r['weights'][method] for r in regions]) if regions else np.zeros((0,784))
    scores=[r['score'] for r in regions];confidence=[r['confidence'] for r in regions];quality=[r['quality'] for r in regions]
    values=calibrated_scores(scores,confidence,quality,config)
    updated,known,residual=fuse(logits,weights,values,gain=config['gain'],method=method)
    return updated,known,residual,weights,values


def predict(public,labels,config,kind,oracle=False,all_rois=False):
    results=[]
    for state,regions in observed_candidates(public,config['history'],kind,all_rois):
        if oracle:
            weights=np.stack([r['weights'][config['mapping']] for r in regions]) if regions else np.zeros((0,784))
            mask=[r['object_id']==labels[state['sample_id']]['target_object_id'] for r in regions]
            updated,known,residual=identity_oracle(state['logits'],weights,mask);values=None
        else:updated,known,residual,weights,values=public_prediction(state,regions,config,kind)
        # Ground truth is accessed only after public scores are computed.
        lab=labels[state['sample_id']];xy=state['bounds'][:2]+state['grid_xy']*(state['bounds'][2:]-state['bounds'][:2])
        distance=np.linalg.norm(xy-lab['target_world'],axis=1);before=state['logits'].argsort(kind='stable')[::-1];after=updated.argsort(kind='stable')[::-1]
        a=bool(distance[before[0]]<=20);b=bool(distance[after[0]]<=20)
        correctcells=distance<=20;rank_before=int(np.flatnonzero(correctcells[before])[0])+1 if correctcells.any() else None
        rank_after=int(np.flatnonzero(correctcells[after])[0])+1 if correctcells.any() else None
        target_instances=[j for j,r in enumerate(regions) if r['object_id']==lab['target_object_id']]
        target_support=(weights[target_instances]>0).any(0) if target_instances else np.zeros(784,bool)
        is_target=(np.array([r['object_id']==lab['target_object_id'] for r in regions]))
        identityrank=None
        if values is not None and target_instances:identityrank=min(int(np.flatnonzero(np.argsort(-values,kind='stable')==j)[0])+1 for j in target_instances)
        metrics=dict(r1=float(b),r4=float((distance[after[:4]]<=20).any()),r8=float((distance[after[:8]]<=20).any()),ne=float(distance[after[0]]),recovery=float(not a and b),damage=float(a and not b),net_recovery=float(b)-float(a),static_error=float(not a),target_observed=float(bool(target_instances)),error_target_observed=float(not a and bool(target_instances)),candidate_coverage=float(known.mean()),rank_change=float(rank_before-rank_after) if rank_before is not None else None,top1_changed=float(before[0]!=after[0]),target_residual=float(residual[correctcells].max()) if correctcells.any() else None,static_top1_residual=float(residual[before[0]]),residual_max=float(np.max(np.abs(residual))),correct_grid_association=float((target_support & correctcells).any()) if target_instances else None,target_instance_r1=float(identityrank==1) if identityrank is not None else None)
        results.append(dict(sample_id=state['sample_id'],family_id=state['family_id'],map_name=state['map_name'],episode_id=state['episode_id'],cutoff_step=state['cutoff_step'],metrics=metrics,static_top1=int(before[0]),updated_top1=int(after[0]),rank_before=rank_before,rank_after=rank_after,static_margin=float(state['logits'][before[0]]-state['logits'][before[1]]),target_object_id=lab['target_object_id'],target_visible_identity_rank=identityrank,weak_eligible=lab['weak_eligible'],observed_object_ids=[r['object_id'] for r in regions],instance_scores=values.tolist() if values is not None else None,instance_confidence=[r.get('confidence') for r in regions],instance_view_counts=[r['views'] for r in regions],target_supported_cell_count=int(target_support.sum()),known_cells=np.flatnonzero(known).tolist(),changed_cells=np.flatnonzero(residual!=0).tolist()))
    return results


def stats(rows):
    result={k:summarize_rows(rows,k,repetitions=2000,seed=SEED) for k in rows[0]['metrics']}
    for k in ['recovery','damage','net_recovery','target_observed','error_target_observed','top1_changed']:
        result[k]['count']=sum(r['metrics'][k] for r in rows)
    eligible=[r for r in rows if r['metrics']['error_target_observed']]
    result['independent_error_visible']=dict(states=len(eligible),families=len({r['family_id'] for r in eligible}),objects=len({(r['map_name'],r['target_object_id']) for r in eligible}),weak_semantic_eligible_states=sum(r['weak_eligible'] for r in eligible))
    return result


def calibrate(prepared,output):
    root=Path(output)
    if root.exists():raise ValueError('no overwrite')
    root.mkdir(parents=True);data=load_prepared(prepared);public=data['public'];labels=data['labels'];trials=[];selected={};diagnostic={}
    protocol=json.loads(Path(__file__).with_name('protocol.json').read_text())
    for method in METHODS:
        candidates=[]
        for mode,gain,confidence,history in itertools.product(protocol['calibration']['score_modes'],protocol['calibration']['gains'],protocol['calibration']['confidence'],protocol['calibration']['history']):
            config=dict(mapping=method,score_mode=mode,gain=gain,confidence=confidence,history=history)
            rows=predict(public,labels,config,'real')
            value=dict(config=config,r1=np.mean([r['metrics']['r1'] for r in rows]),recoveries=sum(r['metrics']['recovery'] for r in rows),damages=sum(r['metrics']['damage'] for r in rows),net=sum(r['metrics']['net_recovery'] for r in rows))
            candidates.append(value)
        key=lambda v:(v['r1'],v['net'],-v['damages'],-v['config']['gain'],-int(v['config']['history']),json.dumps(v['config'],sort_keys=True))
        selected[method]=max(candidates,key=key)['config']
        diagnostic[method]=max([c for c in candidates if c['config']['gain']>0],key=key)['config']
        trials+=candidates
        print(method,'selected',selected[method],'nonzero',diagnostic[method],flush=True)
    write_json(root/'frozen_config.json',dict(frozen=True,seed=SEED,selected=selected,nonzero_diagnostic=diagnostic,prepared_sha256=file_sha256(Path(prepared)/'prepared.pt'),protocol_sha256=file_sha256(Path(__file__).with_name('protocol.json')),selection_split='Train internal Dev only',trials=trials))


def evaluate(prepared,config,output):
    root=Path(output)
    if root.exists():raise ValueError('no overwrite')
    root.mkdir(parents=True);started=time.perf_counter();data=load_prepared(prepared);public=data['public'];labels=data['labels'];cfg=json.loads(Path(config).read_text());assert cfg['frozen']
    outputs={};baseconfig=dict(mapping='centroid',score_mode='raw',gain=0,confidence='learned',history=False)
    outputs['geometry']=predict(public,labels,baseconfig,'real')
    for method in METHODS:
        for suite in ['selected','nonzero_diagnostic']:
            for kind in ['real','zero','shuffled']:
                name=method+'_'+suite+'_'+kind; outputs[name]=predict(public,labels,cfg[suite][method],kind)
        # Original raw/gain025/current remains a fixed reference, not retuned.
        legacy=dict(baseconfig,mapping=method,gain=.025)
        outputs[method+'_legacy_raw_real']=predict(public,labels,legacy,'real')
        for history in [False,True]:
            oracleconfig=dict(baseconfig,mapping=method,history=history)
            for full in [False,True]:
                name=method+'_oracle_'+('history' if history else 'current')+('_all_cached_rois' if full else '_common_features')
                outputs[name]=predict(public,labels,oracleconfig,'real',oracle=True,all_rois=full)
    summaries={};paired={};controls={}
    keys=['r1','r4','r8','ne','net_recovery','rank_change']
    for name,rows in outputs.items():
        write_rows(root/(name+'_predictions.jsonl'),rows);summaries[name]=stats(rows)
        paired[name]={k:paired_difference(outputs['geometry'],rows,k,repetitions=2000,seed=SEED) for k in keys}
    for method in METHODS:
        real=outputs[method+'_selected_real']
        controls[method]={kind:paired_difference(outputs[method+'_selected_'+kind],real,'r1',repetitions=2000,seed=SEED) for kind in ['zero','shuffled']}
    gates={}
    for method in METHODS:
        delta=paired[method+'_selected_real']['r1'];s=summaries[method+'_selected_real'];c=controls[method]
        gates[method]=dict(localization=delta['mean']>=.01 and delta['map_cluster_95ci'][0]>0,real_independent=all(c[k]['map_cluster_95ci'][0]>0 for k in c),safety=s['net_recovery']['count']>0,independent_visible_cases=s['independent_error_visible']['families']>=50,oracle=summaries[method+'_oracle_current_common_features']['recovery']['count']>0)
    write_json(root/'metrics.json',dict(models=summaries,paired_vs_geometry=paired,real_minus_controls=controls,gates=gates,config=cfg,states=len(public),seed=SEED,prepared_sha256=file_sha256(Path(prepared)/'prepared.pt'),seconds=time.perf_counter()-started))
    print(json.dumps(dict(gates=gates,models={n:{k:v[k].get('count',v[k]['mean']) for k in ['r1','recovery','damage','net_recovery']} for n,v in summaries.items()})),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','calibrate','evaluate']);p.add_argument('--cache');p.add_argument('--features');p.add_argument('--controls');p.add_argument('--checkpoint');p.add_argument('--prepared');p.add_argument('--config');p.add_argument('--output',required=True);a=p.parse_args()
    if a.mode=='prepare':prepare(a.cache,a.features,a.controls,a.checkpoint,a.output)
    elif a.mode=='calibrate':calibrate(a.prepared,a.output)
    else:evaluate(a.prepared,a.config,a.output)
if __name__=='__main__':main()
