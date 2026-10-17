"""Acceptance checks; never equate missing fair visual retraining with completion."""
import json,csv
from pathlib import Path
import numpy as np
from graphnav.checkpointing import file_sha256,atomic_json
from scripts.geombelief_map_experiments import OUT,OLD,HERE


def main():
    plan=json.loads((OUT/'frozen_plan.json').read_text())
    for path,sha in plan['input_sha256'].items():assert file_sha256(path)==sha,path
    rows=list(csv.DictReader((OUT/'map_geometry_results.csv').open()));assert len(rows)==72
    assert all(r['status']=='Done' for r in rows)
    for r in rows:assert int(r['samples'])==int(r['expected_samples']) and file_sha256(r['checkpoint'])==r['checkpoint_sha256']
    control=list(csv.DictReader((OUT/'controller_ablation_results.csv').open()));assert len(control)==24 and all(r['status']=='Done' for r in control)
    visual=list(csv.DictReader((OUT/'visual_input_attribution.csv').open()))
    assert all(r['status']=='Done' for r in visual if r['kind']!='fair_retraining')
    assert all(r['status']=='Pending' for r in visual if r['kind']=='fair_retraining')
    navigation_paths={Path(r['predictions']) for r in rows if r['endpoint']=='navigation'}
    navigation_paths|={Path(r['result_path'])/'predictions.jsonl' for r in control}
    navigation_paths|={Path(r['result_path'])/'predictions.jsonl' for r in visual if r['kind']!='fair_retraining'}
    for trace in navigation_paths:
        for line in trace.open():
            row=json.loads(line);poses=np.asarray(row['poses'])
            assert len(poses)-1<=20 and np.isfinite(poses).all(),trace
            if len(poses)>1:
                delta=np.diff(poses,axis=0)
                assert np.linalg.norm(delta[:,:2],axis=1).max()<=50.+1e-4,trace
                assert np.abs(delta[:,2]).max()<=10.+1e-4,trace
    for seed in (0,1,2):
        reference=json.loads((OLD/'training'/f'shared_full_polygon__full_polygon__seed{seed}'/'complete.json').read_text())
        oldhistory=json.loads((OLD/'training'/f'shared_full_polygon__full_polygon__seed{seed}'/'history.json').read_text())
        for path in (OUT/'training').glob(f'*__seed{seed}'):
            done=json.loads((path/'complete.json').read_text());history=json.loads((path/'history.json').read_text())
            assert done['initial_state_sha256']==reference['initial_state_sha256'] and len(history)==15
            assert [r['data_order_sha256'] for r in history]==[r['data_order_sha256'] for r in oldhistory]
    assert (OUT/'controller_context_audit.json').exists() and (OUT/'belief_figure_manifest.json').exists() and (OUT/'shuffle_effectiveness.json').exists()
    for split in ('val_seen','val_unseen'):
        shared={}
        for folder in (OUT/'controller_fixed_goal').glob(f'*/{split}'):
            for line in (folder/'predictions.jsonl').open():
                r=json.loads(line);key=tuple(r['episode_id'])
                assert r['belief_frozen_at_initial_observation']
                if key in shared:assert shared[key]==r['initial_field_sha256']
                else:shared[key]=r['initial_field_sha256']
        assert len(shared)==(2470 if split=='val_seen' else 2697)
    # Formal baseline differences are measured, not silently merged.
    parity=[]
    for split in ('val_seen','val_unseen'):
        old=Path(json.loads((OUT/f'formal_baseline_{split}.json').read_text())['source'])
        oldrows={tuple(r['episode_id']):r for r in [json.loads(l) for l in (old/'predictions.jsonl').read_text().splitlines()]}
        newrows=[json.loads(l) for l in (OUT/'controller/predicted_guarded_blend'/split/'predictions.jsonl').read_text().splitlines()]
        samepath=0;metrics={k:[] for k in ('SR','SPL','OSR','NE')};changed_sr=[]
        for r in newrows:
            before=oldrows[tuple(r['episode_id'])]
            samepath+=int(np.array_equal(np.asarray(r['poses']),np.asarray(before['poses'])))
            for k in metrics:metrics[k].append(r['metrics'][k]-before['metrics'][k])
            if r['metrics']['SR']!=before['metrics']['SR']:changed_sr.append(r['episode_id'])
        parity.append(dict(split=split,episodes=len(newrows),exact_same_trajectories=samepath,metric_deltas={k:float(np.mean(v)) for k,v in metrics.items()},changed_SR_episode_ids=changed_sr,original_source=str(old),paired_control='new matched runtime, not archived reference'))
    atomic_json(parity,OUT/'formal_runtime_parity.json')
    atomic_json(dict(status='Done',map_result_rows=72,controller_result_rows=24,new_training_runs=15,correct_training_reused=3,
        new_map_full_split_evaluations=30,new_formal_control_full_split_evaluations=24,new_visual_frozen_input_full_split_evaluations=6,
        completed_core=True,fair_visual_retraining='Pending, about 8 GPU-hours plus evaluation for three one-seed matched input schemes; not claimed complete',
        original_inputs_and_weights_unchanged=True,all_seeds_initialization_and_data_order_match=True),OUT/'acceptance.json')

if __name__=='__main__':main()
