"""Offline association/rank bottleneck audit; labels never change predictions."""
import argparse,json
from pathlib import Path
import numpy as np
from .evaluate import load_prepared,observed_candidates,public_prediction,write_json,write_rows,SEED
from experiments.bidirectional_core.stats import summarize_rows


def run(prepared,output):
    root=Path(output)
    if root.exists():raise ValueError('no overwrite')
    root.mkdir(parents=True);data=load_prepared(prepared);rows=[];config=dict(mapping='centroid',score_mode='raw',gain=.025,confidence='learned',history=False)
    for state,regions in observed_candidates(data['public'],False,'real'):
        updated,known,residual,weights,values=public_prediction(state,regions,config,'real');lab=data['labels'][state['sample_id']]
        xy=state['bounds'][:2]+state['grid_xy']*(state['bounds'][2:]-state['bounds'][:2]);distance=np.linalg.norm(xy-lab['target_world'],axis=1)
        logits=state['logits'];rank=np.argsort(-logits);correct=distance<=20;top=int(rank[0])
        identity=[j for j,r in enumerate(regions) if r['object_id']==lab['target_object_id']]
        original=None;candidatecorrect=None;collision=None
        if identity:
            original=min(int(np.flatnonzero(np.argsort(-values)==j)[0])+1 for j in identity)
            target_support=(weights[identity]>0).any(0)
            collision=int(((weights>0).sum(0)[target_support]>1).sum())
            candidatecorrect=bool((target_support & correct).any())
        unit=residual/.025;denom=unit-unit[top];eligible=correct & (denom>1e-12)
        required=float(np.min((logits[top]-logits[eligible])/denom[eligible])) if eligible.any() else None
        metrics=dict(static_r1=float(distance[top]<=20),static_r8=float((distance[rank[:8]]<=20).any()),static_r20=float((distance[rank[:20]]<=20).any()),static_margin=float(logits[rank[0]]-logits[rank[1]]),static_range=float(logits.max()-logits.min()),score_range=float(np.ptp([r['score'] for r in regions])) if regions else None,score_std=float(np.std([r['score'] for r in regions])) if regions else None,confidence_mean=float(np.mean([r['confidence'] for r in regions])) if regions else None,confidence_min=float(min(r['confidence'] for r in regions)) if regions else None,confidence_max=float(max(r['confidence'] for r in regions)) if regions else None,legacy_max_abs_residual=float(np.abs(residual).max()),top1_changes=float(updated.argmax()!=top),target_visible=float(bool(identity)),target_identity_r1=float(original==1) if identity else None,target_centroid_hits_success_region=float(candidatecorrect) if candidatecorrect is not None else None,target_colliding_cells=float(collision) if collision is not None else None,roi_count=len(regions),distinct_cells=int((weights.sum(0)>0).sum()),roi_cell_collisions=int((weights>0).sum()-(weights.sum(0)>0).sum()),required_gain_lower_bound_to_beat_static_top1=required)
        rows.append(dict(sample_id=state['sample_id'],family_id=state['family_id'],map_name=state['map_name'],metrics=metrics,weak_eligible=lab['weak_eligible'],target_identity_rank=original,static_error_and_target_visible=bool(distance[top]>20 and identity)))
    summaries={k:summarize_rows(rows,k,repetitions=2000,seed=SEED) for k in rows[0]['metrics']}
    conditional={}
    for name,part in [('static_errors_visible',[r for r in rows if r['static_error_and_target_visible']]),('weak_judged',[r for r in rows if r['weak_eligible']])]:
        conditional[name]=dict(states=len(part),families=len({r['family_id'] for r in part}),metrics={k:summarize_rows(part,k,repetitions=2000,seed=SEED) for k in ['target_identity_r1','target_centroid_hits_success_region','required_gain_lower_bound_to_beat_static_top1','target_colliding_cells']})
    write_rows(root/'predictions.jsonl',rows);write_json(root/'metrics.json',dict(summary=summaries,conditional=conditional,seed=SEED,required_gain_note='necessary bound to beat original static winner, not sufficient to beat other rising candidates; labels are evaluator-only'))
    print(json.dumps(dict(conditional=conditional,range_audit={k:summaries[k] for k in ['static_margin','score_std','confidence_mean','legacy_max_abs_residual']})),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.prepared,a.output)
