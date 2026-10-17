"""Exact rule-stop counterfactual prefixes, certified against actual 256 smoke rollouts.

Until the first additional rule stop, motion/observations/belief are identical to
formal Guarded-Blend. Rule stop can only shorten that executed prefix. It never
requires a future frame or hypothetical unobserved action.
"""
import json,time,subprocess
from pathlib import Path
import numpy as np
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.closedloop import score_trace,aggregate
from graphnav.workspace_paths import PROJECT
from scripts.geombelief_map_experiments import OUT,HERE,init
from scripts.polygon_ablation_navigation import resources
from scripts.geombelief_control_experiments import events

def prefix(row):
    d=row['step_diagnostics'];eligible=[i for i,v in enumerate(d) if v.get('predicted_goal_distance') is not None and v['predicted_goal_distance']<=.001 and v.get('stop_altitude_guard',False)]
    stop=eligible[0] if eligible else None
    return (row['poses'][:stop+1],d[:stop+1],row['decision_seconds'][:stop+1],'policy_stop') if stop is not None else (row['poses'],d,row['decision_seconds'],row['termination'])

def main():
    plan=init()
    # Wait for BOTH independently executed checks, not for a statistical outcome.
    for split in ('val_seen','val_unseen'):
        while not (OUT/'controller/rule_arrival'/f'{split}_smoke128/metrics.json').exists():time.sleep(30)
    saved={};proof=[]
    for split in ('val_seen','val_unseen'):
        source=PROJECT/'GraphNav/runs/graphnav_v1_stage1'/('full_seen' if split=='val_seen' else 'full_unseen')/'predictions.jsonl'
        original=[json.loads(l) for l in source.read_text().splitlines()];saved[split]=(source,original)
        mapping={tuple(r['episode_id']):r for r in original}
        smoke=[json.loads(l) for l in (OUT/'controller/rule_arrival'/f'{split}_smoke128/predictions.jsonl').read_text().splitlines()]
        assert len(smoke)==128
        for r in smoke:
            poses,d,t,term=prefix(mapping[tuple(r['episode_id'])])
            if not np.array_equal(np.asarray(poses),np.asarray(r['poses'])) or term!=r['termination']:
                atomic_json(dict(status='Not applicable',reason='Formal-prefix parity failed; live full evaluation remains authoritative',episode_id=r['episode_id'],split=split),OUT/'rule_replay_certificate.json');return
        proof.append(dict(split=split,smoke_episodes=128,all_trajectories_exact=True,all_terminations_exact=True,source_sha256=file_sha256(source)))
    atomic_json(dict(status='Verified',proof=proof,kind='Exact causal counterfactual prefix; actual 256 smoke rollouts matched',not_inference_timing=True),OUT/'rule_replay_certificate.json')
    for split in ('val_seen','val_unseen'):
        folder=OUT/'controller/rule_arrival'/split
        if folder.exists():
            # Never race or overwrite an already started full evaluation.
            atomic_json(dict(status='Skipped',reason='Full output already exists'),OUT/f'rule_replay_{split}_skip.json');continue
        data,catalog,base,controller,protocol,ground=resources(plan,split)
        source,original=saved[split];assert len(original)==len(data)
        temporary=OUT/'controller/rule_arrival'/(split+'_replay_build');temporary.mkdir(exist_ok=True)
        if (temporary/'predictions.jsonl').exists():raise ValueError('Preserve existing interrupted replay; inspect before resume')
        signature=dict(arm='rule_arrival',split=split,smoke=False,checkpoint_sha256=file_sha256(plan['base_checkpoint']),
            source_sha256=file_sha256(HERE/'scripts/geombelief_control_experiments.py'),policy_sha256=file_sha256(HERE/'graphnav/geombelief_control.py'),stop_threshold=.3,arrival_guard_m=.001,protocol=protocol.__dict__,visual='original checkpoint/current+checkpoint history',candidate_selection='always original belief top1; oracle replaces only its controller coordinate; unchanged candidate context',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
        atomic_json(signature,temporary/'config.json')
        manifest=[dict(index=i,episode_id=list(data[i].episode_id),map_name=data[i].map_name) for i in range(len(data))];atomic_json(manifest,temporary/'episodes.json')
        metrics=[]
        with (temporary/'predictions.jsonl').open('w') as f:
            for i,row in enumerate(original):
                record=data[i];assert row['episode_id']==list(record.episode_id)
                poses,diag,timings,term=prefix(row)
                score=score_trace(np.asarray(poses),record.target_position,record.trajectory,term,decision_seconds=[])
                for d in diag:d.update(control_arm='rule_arrival',oracle_diagnostic=False,learned_stop_requested=d['stop_probability']>=.3,predicted_arrival=d['predicted_goal_distance']<=.001,arrival_guard_m=.001,success_radius_m=20.)
                # Timing is explicitly inherited reference timing, not profiled rule latency.
                result=dict(index=i,episode_id=row['episode_id'],map_name=row['map_name'],poses=poses,termination=term,step_diagnostics=diag,decision_seconds=timings,metrics=score,events=events(poses,diag,record.target_position,term),evaluation_kind='certified_counterfactual_prefix',timings_are_reference=True)
                f.write(json.dumps(result)+'\n');metrics.append(score)
        atomic_json(dict(status='Done',signature=signature,metrics=aggregate(metrics),evaluation_kind='certified_counterfactual_prefix',source_trace=str(source),source_sha256=file_sha256(source),certificate=str(OUT/'rule_replay_certificate.json'),timings_not_profiled=True),temporary/'metrics.json')
        temporary.rename(folder)
if __name__=='__main__':main()
