"""Actual cached-state diagnostics; oracle target access is evaluation-only.

No experimental decision or threshold is selected on Seen. Oracle upper bounds
are never callable policies or model inputs.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from experiments.bidirectional_core.cache import CachedCoreDataset,jsonl
from experiments.bidirectional_core.stats import summarize_rows,paired_difference
from graphnav.checkpointing import file_sha256


def run(args):
    root=Path(args.output)
    if root.exists():raise ValueError('refusing diagnostic overwrite')
    root.mkdir(parents=True)
    dataset=CachedCoreDataset(args.cache,'E1')
    episodes={r['episode_id']:r for r in jsonl(args.episodes)}
    rows=[];history={}
    order=sorted(range(len(dataset)),key=lambda i:(dataset[i]['metadata']['episode_id'],dataset[i]['metadata']['cutoff_step']))
    for i in order:
        row=dataset[i];p=row['public'];m=row['metadata'];lab=row['labels']
        grid=p['bounds'][:2]+p['grid_xy']*(p['bounds'][2:]-p['bounds'][:2])
        goal=lab['target_world'][:2];distance=torch.linalg.vector_norm(grid-goal,dim=-1)
        logits=p['static_logits'];rank=logits.argsort(descending=True);target_ids=(distance<=20).nonzero().ravel()
        target_id=str(episodes[m['episode_id']]['private_labels']['target_object_id'])
        target_roi=[j for j,r in enumerate(m['roi_objects']) if r['object_id']==target_id and p['roi_valid'][j]]
        observed_ids=p['roi_grid_ids'][p['roi_valid']].unique()
        target_centers=[int(p['roi_grid_ids'][j]) for j in target_roi]
        memory=history.setdefault(m['episode_id'],{})
        for j,region in enumerate(m['roi_objects']):
            if p['roi_valid'][j]:memory[region['object_id']]=dict(grid_id=int(p['roi_grid_ids'][j]),cutoff_step=m['cutoff_step'])
        if any(v['cutoff_step']>m['cutoff_step'] for v in memory.values()):raise ValueError('future history evidence')
        historical_target=memory.get(target_id)
        center_errors=[float(distance[j]) for j in target_centers]
        grid_ids=p['roi_grid_ids'][p['roi_valid']]
        scores=p['roi_features'].float()@p['native_text'].float() if len(p['roi_features']) else torch.empty(0)
        static_correct=bool(distance[rank[0]]<=20)
        # Perfect region-identity evidence with existing center-only association.
        oracle_center=logits.clone()
        if target_centers:oracle_center[target_centers]=logits.max()+20
        # Ideal correctly observed goal cell, diagnostic ceiling only. Only allow
        # it when the true target object has an actual valid current observation.
        oracle_goal=logits.clone()
        if target_roi and len(target_ids):oracle_goal[target_ids]=logits.max()+20
        oracle_history=logits.clone()
        if historical_target:oracle_history[historical_target['grid_id']]=logits.max()+20
        sem_valid=lab['semantic_valid_mask'] & p['roi_valid'];positive=lab['positive_roi_mask'] & p['roi_valid']
        usable=bool(positive.any() and (sem_valid & ~positive).any())
        native_correct=None
        if usable:
            selected=scores.masked_fill(~sem_valid,-1e4).argmax()
            native_correct=float(positive[selected])
        r=dict(sample_id=m['sample_id'],family_id=m['family_id'],map_name=m['map_name'],direction=m['direction'],
            static_r1=float(static_correct),static_r4=float((distance[rank[:4]]<=20).any()),static_r8=float((distance[rank[:8]]<=20).any()),
            static_ne=float(distance[rank[0]]),valid_roi_count=int(p['roi_valid'].sum()),distinct_observed_grid_cells=len(observed_ids),
            collision_count=len(grid_ids)-len(observed_ids),actual_target_roi_visible=float(bool(target_roi)),
            center_distance_m=min(center_errors) if center_errors else None,
            observed_target_center_xy20=float(any(d<=20 for d in center_errors)) if target_roi else None,
            target_center_in_a3_top8=float(any(j in rank[:8] for j in target_centers)) if target_roi else None,
            candidate_upperbound_r1=float(bool((distance[observed_ids]<=20).any())) if len(observed_ids) else None,
            oracle_region_center_r1=float(distance[oracle_center.argmax()]<=20),
            oracle_observed_goal_cell_r1=float(distance[oracle_goal.argmax()]<=20),native_weak_r1=native_correct,
            history_target_roi_visible=float(historical_target is not None),oracle_history_region_center_r1=float(distance[oracle_history.argmax()]<=20),
            static_top1_margin=float(logits[rank[0]]-logits[rank[1]]),native_cosine_std=float(scores[p['roi_valid']].std(unbiased=False)) if p['roi_valid'].any() else None,
            target_original_native_min_side=min((m['roi_objects'][j]['native_min_side'] for j in target_roi),default=None))
        for kind in ['oracle_region_center','oracle_observed_goal_cell']:
            correct=bool(r[kind+'_r1']);r[kind+'_recovery']=float(correct) if not static_correct else None
            r[kind+'_damage']=float(not correct) if static_correct else None
        rows.append(r)
    with (root/'predictions.jsonl').open('w') as h:
        for row in rows:h.write(json.dumps(row)+'\n')
    keys=[k for k in rows[0] if k not in {'sample_id','family_id','map_name','direction'}]
    statistical=[dict(sample_id=r['sample_id'],family_id=r['family_id'],map_name=r['map_name'],direction=r['direction'],metrics={k:r[k] for k in keys}) for r in rows]
    summary={k:summarize_rows(statistical,k,repetitions=2000) for k in keys}
    summary['by_direction']={direction:{k:summarize_rows([r for r in statistical if r['direction']==direction],k,repetitions=2000) for k in keys} for direction in ['forward','reverse']}
    summary['provenance']=dict(cache_manifest_sha256=file_sha256(Path(args.cache)/'manifest.json'),episodes_sha256=file_sha256(args.episodes),
        oracle_diagnostic_only=True,actual_states=len(rows),protocol_selected_on_seen=False,
        region_center_ambiguity='identity oracle can choose wrong target position for a large object; centroid association does not model language within-object offset')
    (root/'metrics.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps({k:summary[k] for k in ['actual_target_roi_visible','observed_target_center_xy20','oracle_region_center_r1','oracle_observed_goal_cell_r1','native_weak_r1']},indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cache',required=True);parser.add_argument('--episodes',required=True);parser.add_argument('--output',required=True)
    run(parser.parse_args())


if __name__=='__main__':main()
