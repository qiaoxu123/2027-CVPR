"""Evaluate final matched heads on one immutable Seen cache, without RGB inference.

Identity/geometry supervision is explicitly distinguished from human-verified
appearance semantics. Checkpoint selection and interventions are fixed before scores.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from .model import CoreModel,collate_core
from .stats import summarize_rows,paired_difference


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def episode_lookup(rows):
    result={}
    for row in rows:
        key=row['episode_id']
        if key in result:raise ValueError('duplicate private evaluation episode identity')
        result[key]=row
    return result


def audit_episode(metadata,episodes):
    episode=episodes[metadata['episode_id']]
    for key in ['family_id','map_name','direction','source_episode_index']:
        if key in metadata and metadata[key]!=episode[key]:raise ValueError('private episode/cache identity mismatch: '+key)
    return episode


def pixel_group(minimum):
    if minimum is None:return 'unknown'
    if minimum<8:return 'below8'
    if minimum<24:return 'low_8_23'
    if minimum<48:return 'medium_24_47'
    return 'high_48_plus'


def object_metadata(metadata,n):
    objects=metadata.get('roi_objects',[])
    if len(objects)!=n:raise ValueError('ROI metadata length differs from actual features')
    result=[]
    for i,obj in enumerate(objects):
        if 'object_id' not in obj:raise ValueError('missing private audit object identity')
        obj=dict(obj);obj['object_id']=str(obj['object_id']);obj['object_type']=obj.get('object_type',obj.get('type'))
        if obj['object_type'] is None:raise ValueError('missing ROI object type')
        size=obj.get('roi_pixel_size',obj.get('native_pixel_size'))
        obj['native_min_side']=obj.get('native_min_side',min(size) if size else None)
        result.append(obj)
    return result


def audit_row(row):
    if set(row['public'])!=CoreModel.PUBLIC_KEYS:raise ValueError('public fields differ; refusing private/extra inputs')
    metadata=row['metadata'];cutoff=int(metadata['cutoff_step'])
    if int(metadata.get('current_observation_step',cutoff))!=cutoff:raise ValueError('current observation/cutoff mismatch')
    objects=object_metadata(metadata,len(row['public']['roi_grid_ids']))
    for obj in objects:
        if int(obj['observed_step'])>cutoff or int(obj['observation_cutoff_step'])>cutoff:raise ValueError('future ROI evidence forbidden')
    return objects


def object_rank(scores,valid,positive,judged,objects, *, same_class=False):
    eligible=np.asarray(valid,dtype=bool)&np.asarray(judged,dtype=bool)
    positives=np.asarray(positive,dtype=bool)&eligible
    if not positives.any():return None
    types={objects[i]['object_type'] for i in np.flatnonzero(positives)}
    if same_class:eligible&=np.array([o['object_type'] in types for o in objects])
    grouped=defaultdict(list);labels={}
    for i in np.flatnonzero(eligible):
        oid=objects[i]['object_id'];grouped[oid].append(float(scores[i]));label=bool(positives[i])
        if oid in labels and labels[oid]!=label:raise ValueError('same object has conflicting judged identity labels')
        labels[oid]=label
    if len(grouped)<2 or not any(labels.values()) or all(labels.values()):return None
    ranked=sorted(grouped,key=lambda k:(-np.mean(grouped[k]),k));positive_scores=[np.mean(grouped[k]) for k in grouped if labels[k]];negative_scores=[np.mean(grouped[k]) for k in grouped if not labels[k]]
    return dict(r1=float(labels[ranked[0]]),r5=float(any(labels[k] for k in ranked[:5])),
                pairwise=float(np.mean([p>n for p in positive_scores for n in negative_scores])),
                candidate_objects=len(grouped),positive_objects=sum(labels.values()),top_object_id=ranked[0],
                rankings=[dict(object_id=k,score=float(np.mean(grouped[k])),positive=labels[k]) for k in ranked])


def swap_plan(valid,positive,judged,objects,preferred_positive_id=None):
    valid=np.asarray(valid,bool);positive=np.asarray(positive,bool);judged=np.asarray(judged,bool)
    positions=defaultdict(list)
    for i in np.flatnonzero(valid&judged):positions[objects[i]['object_id']].append(int(i))
    for po in ([str(preferred_positive_id)] if preferred_positive_id is not None else sorted(positions)):
        if po not in positions:continue
        pi=positions[po]
        if not all(positive[pi]):continue
        for no in sorted(positions):
            ni=positions[no]
            if no==po or any(positive[ni]) or objects[pi[0]]['object_type']!=objects[ni[0]]['object_type']:continue
            # Fixed source ROI chosen before any scores. Actual vectors, not synthetic attributes.
            return dict(positive_object_id=po,negative_object_id=no,positive_indices=pi,negative_indices=ni,
                        positive_source_index=pi[0],negative_source_index=ni[0])
    return None


def swap_outcomes(original_scores,swap_scores,valid,positive,judged,objects,relocated_positive):
    original=object_rank(original_scores,valid,positive,judged,objects)
    fixed=object_rank(swap_scores,valid,positive,judged,objects)
    source=object_rank(swap_scores,valid,relocated_positive,judged,objects)
    return dict(original=original,fixed_labels=fixed,source_tracking=source,
                real_minus_fixed=original['r1']-fixed['r1'] if original and fixed else None)


def pair_outcomes(original_scores,swap_scores,valid,positive,judged,objects,plan):
    pair_judged=np.zeros(len(objects),dtype=bool)
    pair_judged[plan['positive_indices']+plan['negative_indices']]=True
    pair_judged &= np.asarray(judged,bool)
    relocated=np.zeros(len(objects),dtype=bool);relocated[plan['negative_indices']]=True
    result=swap_outcomes(original_scores,swap_scores,valid,positive,pair_judged,objects,relocated)
    real_margin=float(np.mean(np.asarray(original_scores)[plan['positive_indices']])-np.mean(np.asarray(original_scores)[plan['negative_indices']]))
    swap_margin=float(np.mean(np.asarray(swap_scores)[plan['positive_indices']])-np.mean(np.asarray(swap_scores)[plan['negative_indices']]))
    result.update(real_margin=real_margin,swap_margin=swap_margin,real_tie=abs(real_margin)<=1e-8)
    return result


def load_model(checkpoint,manifest,device):
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if state.get('epoch')!=3:raise ValueError('only the predeclared completed epoch-3 checkpoint is evaluable')
    protocol=state['protocol']
    shared=set(protocol['frozen_resources']) & set(manifest['frozen_resources'])
    if not shared:raise ValueError('no shared pinned frozen backbone resources')
    if any(protocol['frozen_resources'][p]!=manifest['frozen_resources'][p] for p in shared):raise ValueError('Train/Seen frozen resource mismatch')
    model=CoreModel(hidden=int(protocol['hidden']),top_k=int(protocol['top_k'])).to(device)
    model.load_state_dict(state['model'],strict=True);model.eval().requires_grad_(False)
    return model,state


@torch.inference_mode()
def evaluate_condition(dataset,checkpoint,manifest,device,batch_size,episodes):
    model,state=load_model(checkpoint,manifest,device);records=[];started=time.perf_counter()
    for offset in range(0,len(dataset),batch_size):
        rows=[dataset[i] for i in range(offset,min(offset+batch_size,len(dataset)))];audited=[audit_row(row) for row in rows];[audit_episode(row['metadata'],episodes) for row in rows];public,labels,metadata=collate_core(rows)
        public={k:v.to(device) for k,v in public.items()}
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):output=model(public)
        if not torch.isfinite(output.belief_logits).all() or not torch.isfinite(output.roi_logits).all():raise ValueError('nonfinite prediction')
        if not torch.equal(output.belief_logits[~output.observed],public['static_logits'].float()[~output.observed]):raise ValueError('unobserved grid score changed')
        plans=[];swapped={k:v.clone() if k=='roi_features' else v for k,v in public.items()};relocated=labels['positive_roi_mask'].clone()
        for i,(row,meta) in enumerate(zip(rows,metadata)):
            n=len(row['public']['roi_grid_ids']);objects=audited[i]
            private_target=episodes[meta['episode_id']]['private_labels'].get('target_object_id')
            plan=swap_plan(output.roi_valid[i,:n].cpu().numpy(),labels['positive_roi_mask'][i,:n].numpy(),labels['semantic_valid_mask'][i,:n].numpy(),objects,preferred_positive_id=private_target) if private_target is not None else None
            plans.append((objects,plan))
            if plan:
                pids,nids=plan['positive_indices'],plan['negative_indices']
                swapped['roi_features'][i,pids]=public['roi_features'][i,plan['negative_source_index']]
                swapped['roi_features'][i,nids]=public['roi_features'][i,plan['positive_source_index']]
                relocated[i,pids]=False;relocated[i,nids]=True
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):swap_output=model(swapped)
        if not torch.isfinite(swap_output.roi_logits).all() or not torch.isfinite(swap_output.belief_logits).all():raise ValueError('nonfinite swapped prediction')
        if not torch.equal(swap_output.belief_logits[~swap_output.observed],public['static_logits'].float()[~swap_output.observed]):raise ValueError('swap changed unknown scores')
        selected,mode,waypoint,stop=output.trajectory.select()
        for i,(row,meta,(objects,plan)) in enumerate(zip(rows,metadata,plans)):
            n=len(objects);valid=output.roi_valid[i,:n].cpu().numpy();pos=labels['positive_roi_mask'][i,:n].numpy();judged=labels['semantic_valid_mask'][i,:n].numpy();scores=output.roi_logits[i,:n].float().cpu().numpy()
            rank=object_rank(scores,valid,pos,judged,objects);hard=object_rank(scores,valid,pos,judged,objects,same_class=True)
            swap_scores=swap_output.roi_logits[i,:n].float().cpu().numpy()
            swap_result=swap_outcomes(scores,swap_scores,valid,pos,judged,objects,relocated[i,:n].numpy()) if plan else None
            swaprank=swap_result['source_tracking'] if swap_result else None
            fixed_swaprank=swap_result['fixed_labels'] if swap_result else None
            pair_result=pair_outcomes(scores,swap_scores,valid,pos,judged,objects,plan) if plan else None
            bounds=public['bounds'][i].float();grid=bounds[:2]+public['grid_xy'][i].float()*(bounds[2:]-bounds[:2]);target=labels['target_world'][i,:2].to(device)
            static_id=int(public['static_logits'][i].argmax());belief_id=int(output.belief_logits[i].argmax());swap_id=int(swap_output.belief_logits[i].argmax())
            distance=lambda index:float((grid[index]-target).norm())
            baseline=float(distance(static_id)<=20);reranked=float(distance(belief_id)<=20)
            positive_sizes=[o['native_min_side'] for j,o in enumerate(objects) if pos[j] and valid[j] and o['native_min_side'] is not None];minimum=max(positive_sizes) if positive_sizes else None
            episode=episodes[meta['episode_id']]
            target_id=episode['private_labels'].get('target_object_id')
            target_id=str(target_id) if target_id is not None else None
            target_sizes=[o['native_min_side'] for j,o in enumerate(objects) if target_id is not None and o['object_id']==target_id and valid[j] and o['native_min_side'] is not None]
            native_target_size=max(target_sizes) if target_sizes else None
            metrics=dict(text_region_r1=rank['r1'] if rank else None,text_region_r5=rank['r5'] if rank else None,
                         same_class_hard_r1=hard['r1'] if hard else None,same_class_pairwise=hard['pairwise'] if hard else None,
                         rgb_pair_real_r1=pair_result['original']['r1'] if pair_result else None,
                         rgb_pair_fixed_swap_r1=pair_result['fixed_labels']['r1'] if pair_result else None,
                         rgb_pair_real_minus_fixed_swap_r1=pair_result['real_minus_fixed'] if pair_result else None,
                         rgb_pair_source_tracking_r1=pair_result['source_tracking']['r1'] if pair_result else None,
                         rgb_pair_real_tie=float(pair_result['real_tie']) if pair_result else None,
                         rgb_pair_real_score_margin=pair_result['real_margin'] if pair_result else None,
                         rgb_real_r1_on_swap_subset=rank['r1'] if rank and fixed_swaprank else None,
                         rgb_swap_fixed_label_r1=fixed_swaprank['r1'] if fixed_swaprank else None,
                         rgb_real_minus_fixed_label_swap_r1=swap_result['real_minus_fixed'] if swap_result else None,
                         rgb_swap_source_tracking_r1=swaprank['r1'] if swaprank else None,
                         rgb_swap_top_object_changed=float(swaprank['top_object_id']!=rank['top_object_id']) if swaprank and rank else None,
                         static_r1_xy20m=baseline,reranked_r1_xy20m=reranked,reranking_delta=reranked-baseline,
                         recovery=reranked if not baseline else None,damage=1-reranked if baseline else None,
                         rgb_swap_reranked_r1_xy20m=float(distance(swap_id)<=20) if plan else None,
                         target_ne_xy_m=distance(belief_id),known_grid_fraction=float(output.observed[i].float().mean()))
            metrics['weak_attribute_text_region_r1']=rank['r1'] if rank and meta.get('direction')=='forward' else None
            metrics['geometry_text_region_r1']=rank['r1'] if rank and meta.get('direction')=='reverse' else None
            eligible=bool(meta.get('static_language_eligible',True));metrics['eligible_static_r1_xy20m']=baseline if eligible else None;metrics['eligible_reranked_r1_xy20m']=reranked if eligible else None
            wp=waypoint[i].float().cpu().numpy();teacher=labels['teacher_world'][i,0].numpy();metrics['first_waypoint_xy_error_m']=float(np.linalg.norm(wp[:2]-teacher[:2]))
            from graphnav.trajectory_policy import calibrated_stop_probability
            stop_prob=float(calibrated_stop_probability(stop[i],20));metrics['prefix_stop_probability']=stop_prob
            record=dict(sample_id=meta.get('sample_id',f'cache:{offset+i}'),family_id=meta['family_id'],map_name=meta['map_name'],direction=meta.get('direction','unknown'),
                        episode_id=meta.get('episode_id'),cutoff_step=meta.get('cutoff_step'),pixel_group=pixel_group(native_target_size),original_target_id=target_id,native_target_min_side=native_target_size,
                        best_multi_positive_pixel_group=pixel_group(minimum),native_positive_best_min_side=minimum,
                        static_language_eligible=eligible,label_status=('language_weak_attribute_binding' if meta.get('direction')=='forward' else 'verified_language_geometry_relation_weak_projected_polygon'),metrics=metrics,
                        roi_count=n,roi_object_ids=[o['object_id'] for o in objects],roi_object_types=[o['object_type'] for o in objects],roi_scores=scores.tolist(),positive_roi_mask=pos.tolist(),semantic_valid_mask=judged.tolist(),roi_valid=valid.tolist(),swap_roi_scores=swap_output.roi_logits[i,:n].float().cpu().tolist(),swap_positive_roi_mask=relocated[i,:n].tolist(),
                        predicted_object_id=rank['top_object_id'] if rank else None,real_correct=rank['r1'] if rank else None,swapped_correct=fixed_swaprank['r1'] if fixed_swaprank else None,swapped_source_tracking_correct=swaprank['r1'] if swaprank else None,
                        judged_objects=rank['candidate_objects'] if rank else 0,rankings=rank,hard_rankings=hard,rgb_swap_plan=plan,rgb_swap_rankings=fixed_swaprank,rgb_swap_source_tracking_rankings=swaprank,rgb_pair_outcomes=pair_result,
                        static_grid_id=static_id,reranked_grid_id=belief_id,unknown_scores_exactly_preserved=True,first_waypoint_world=wp.tolist(),stop_probability=stop_prob)
            records.append(record)
        if (offset//batch_size)%20==0:print(json.dumps(dict(encoded_states=len(records),total=len(dataset),seconds=time.perf_counter()-started)),flush=True)
    if device.type=='cuda':torch.cuda.synchronize(device)
    return records,dict(seconds=time.perf_counter()-started,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),epoch=state['epoch'],condition=state['protocol']['condition'],visual_gain=float(model.visual_gain),state_count=len(records))


def summarize(records,repetitions, *, episode_views=True):
    keys=sorted({k for row in records for k in row['metrics']});result={k:summarize_rows(records,k,repetitions=repetitions) for k in keys}
    result['groups']={axis:{group:{k:summarize_rows([r for r in records if r[axis]==group],k,repetitions=repetitions) for k in keys} for group in sorted({r[axis] for r in records})} for axis in ['direction','pixel_group']}
    judged_objects={(r['map_name'],x['object_id']) for r in records if r['rankings'] for x in r['rankings']['rankings']}
    positive_objects={(r['map_name'],x['object_id']) for r in records if r['rankings'] for x in r['rankings']['rankings'] if x['positive']}
    result['coverage']=dict(states=len(records),independent_episodes=len({r['episode_id'] for r in records}),independent_families=len({r['family_id'] for r in records}),
                            observed_object_identities=len({(r['map_name'],oid) for r in records for oid in r.get('roi_object_ids',[])}),
                            judged_object_identities=len(judged_objects),positive_object_identities=len(positive_objects),
                            roi_language_eligible=sum(r['metrics']['text_region_r1'] is not None for r in records),
                            same_class_eligible=sum(r['metrics']['same_class_hard_r1'] is not None for r in records),rgb_swap_eligible=sum(r['rgb_swap_plan'] is not None for r in records),
                            independently_verified_attribute_states=0)
    if episode_views:
        by_episode=defaultdict(list)
        for r in records:by_episode[r['episode_id']].append(r)
        first=[];average=[]
        for episode,rows in sorted(by_episode.items()):
            initial=min(rows,key=lambda r:(r['cutoff_step'],r['sample_id']));first.append(initial)
            row=dict(initial);row['metrics']={key:float(np.mean([r['metrics'][key] for r in rows if r['metrics'][key] is not None])) if any(r['metrics'][key] is not None for r in rows) else None for key in keys};average.append(row)
        result['episode_first_prefix']={key:summarize_rows(first,key,repetitions=repetitions) for key in keys}
        result['episode_mean_available_prefix']={key:summarize_rows(average,key,repetitions=repetitions) for key in keys}
        result['episode_note']='First prefix is fixed earliest causal step; mean averages available prefix metrics, not oracle best-view or navigation success.'
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',required=True);p.add_argument('--checkpoint',action='append',required=True,help='E0=/path/epoch03.pt (repeat)');p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');p.add_argument('--batch-size',type=int,default=32);p.add_argument('--bootstrap',type=int,default=2000);p.add_argument('--episodes',default='runs/bidirectional_core_test/data/val_seen/pilot_episodes.jsonl')
    args=p.parse_args();out=Path(args.output)
    if (out/'metrics.json').exists():raise ValueError('refusing to overwrite evaluation')
    from .cache import CachedCoreDataset
    cache=Path(args.cache);manifest=json.loads((cache/'manifest.json').read_text())
    if not manifest.get('complete'):raise ValueError('incomplete frozen Seen cache')
    episodes=episode_lookup(json.loads(line) for line in Path(args.episodes).read_text().splitlines() if line.strip())
    dataset=CachedCoreDataset(cache,condition='E1');out.mkdir(parents=True,exist_ok=True);device=torch.device(args.device)
    populations={};metrics={};provenance={};cache_sha=sha(cache/'manifest.json')
    for item in args.checkpoint:
        condition,path=item.split('=',1)
        if condition not in ['E0','E1','E2'] or condition in populations:raise ValueError('invalid/duplicate condition')
        records,meta=evaluate_condition(dataset,path,manifest,device,args.batch_size,episodes)
        if meta['condition']!=condition:raise ValueError('condition/checkpoint mismatch')
        populations[condition]=records;provenance[condition]=meta;metrics[condition]=summarize(records,args.bootstrap)
        (out/(condition+'_predictions.jsonl')).write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records))
    if sha(cache/'manifest.json')!=cache_sha:raise ValueError('cache manifest changed during evaluation')
    pairs={}
    for left,right in [('E0','E1'),('E1','E2'),('E0','E2')]:
        if left in populations and right in populations:
            keys=sorted({k for r in populations[left] for k in r['metrics']})
            pairs[left+'_to_'+right]={k:paired_difference(populations[left],populations[right],k,repetitions=args.bootstrap) for k in keys}
            first={}
            for condition in [left,right]:
                by_episode=defaultdict(list)
                for row in populations[condition]:by_episode[row['episode_id']].append(row)
                first[condition]=[min(rows,key=lambda r:(r['cutoff_step'],r['sample_id'])) for rows in by_episode.values()]
            pairs[left+'_to_'+right]['episode_first_prefix']={k:paired_difference(first[left],first[right],k,repetitions=args.bootstrap) for k in keys}
            pairs[left+'_to_'+right]['groups']={axis:{group:{k:paired_difference([r for r in populations[left] if r[axis]==group],[r for r in populations[right] if r[axis]==group],k,repetitions=args.bootstrap) for k in keys} for group in sorted({r[axis] for r in populations[left]})} for axis in ['direction','pixel_group']}
    result=dict(protocol='fixed Seen cached states; final epoch3 only',private_episode_labels_sha256=sha(args.episodes),cache_manifest_sha256=cache_sha,cache_manifest_path=str((cache/'manifest.json').resolve()),models=metrics,paired_differences=pairs,provenance=provenance,
                limitations=['no independently human-verified appearance labels','RGBswap chooses evaluator-only original target positive and fixed lexicographic same-class known negative; other positives remain positives in full gallery and are excluded, not relabeled, from the separate two-object pair; transplantation of actual cached image features with fixed destination geometry and quality; not an antonym-label experiment','primary pixel groups use original target actual observed native minside via private evaluator-only episode association; unobserved target Unknown; multi-positive best size is supplemental; confounded object type/map/visibility','state IID intervals optimistic; family/map clustering also reported','offline prefix static localization is not navigation success'])
    (out/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(provenance))

if __name__=='__main__':main()
