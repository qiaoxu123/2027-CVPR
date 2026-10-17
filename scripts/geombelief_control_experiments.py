"""Frozen complete C model control attribution; real current-pose RGB, resumable."""
import json,os,sys,time,fcntl,traceback,subprocess
from pathlib import Path
import numpy as np
import torch
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.closedloop import PublicEpisode,rollout,score_trace,aggregate
from graphnav.geombelief_control import ARMS,controlled_predict_field
from graphnav.a3_runtime import A3TrajectoryPolicy
from graphnav.stage1_runtime import GraphNavStage1Policy
from graphnav.dynamic_runtime import NativeObservationBuilder
from graphnav.workspace_paths import WORKSPACE,PROJECT
from scripts.geombelief_map_experiments import OUT,OLD,init
from scripts.polygon_ablation_navigation import resources


def events(poses,diagnostics,target,termination):
    distance=np.linalg.norm(np.asarray(poses)[:,:2]-np.asarray(target)[:2],axis=-1)
    return dict(true_goal_success_region_steps=np.flatnonzero(distance<=20).tolist(),
        predicted_goal_arrival_steps=[i for i,d in enumerate(diagnostics) if d.get('predicted_goal_distance') is not None and d['predicted_goal_distance']<=.001],
        active_stop=termination=='policy_stop',horizon=termination=='horizon',stagnation=termination=='stagnation',
        initial_predicted_goal_error_m=float(np.linalg.norm(np.asarray(diagnostics[0]['selected_goal_xy'])-np.asarray(target)[:2])),
        final_predicted_goal_error_m=float(np.linalg.norm(np.asarray(diagnostics[-1]['selected_goal_xy'])-np.asarray(target)[:2])))

def main():
    plan=init();lock=(OUT/'control_run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(1)
    atomic_json(dict(pid=os.getpid(),started=time.time(),argv=sys.argv),OUT/'control_process.json')
    A3TrajectoryPolicy.predict_field=controlled_predict_field
    runtime=GraphNavStage1Policy(plan['base_checkpoint'],OUT/'controller_inputs')
    assert runtime.args.grid_size==28 and runtime.stop_threshold==.3
    for module in (runtime.model,runtime.bert.model,runtime.native.model):assert not any(p.requires_grad for p in module.parameters())
    for smoke in (True,False):
        for split in ('val_seen','val_unseen'):
            data,catalog,base,controller,protocol,ground=resources(plan,split)
            from sbfnav.validation import evenly_spaced_indices
            baseline=PROJECT/'GraphNav/runs/graphnav_v1_stage1'/('full_seen' if split=='val_seen' else 'full_unseen')
            if split=='val_seen' and not (baseline/'predictions.jsonl').exists():
                raise FileNotFoundError(baseline)
            # Correct unmodified control is archived, never silently rerun.
            atomic_json(dict(status='Done',source=str(baseline),checkpoint_sha256=file_sha256(plan['base_checkpoint']),
                predictions_sha256=file_sha256(baseline/'predictions.jsonl')),OUT/f'formal_baseline_{split}.json')
            builder=NativeObservationBuilder(base,WORKSPACE/'DATA/sbfnav-data/rgbd',OUT/'controller_observations'/split)
            indices=evenly_spaced_indices(len(data),128 if smoke else None)
            for arm in ARMS:
                folder=OUT/'controller'/arm/(split+('_smoke128' if smoke else ''));folder.mkdir(parents=True,exist_ok=True)
                signature=dict(arm=arm,split=split,smoke=smoke,checkpoint_sha256=file_sha256(plan['base_checkpoint']),
                    source_sha256=file_sha256(Path(__file__)),policy_sha256=file_sha256(HERE/'graphnav/geombelief_control.py'),
                    stop_threshold=.3,arrival_guard_m=.001,protocol=protocol.__dict__,visual='original checkpoint/current+checkpoint history',
                    candidate_selection='always original belief top1; oracle replaces only its controller coordinate; unchanged candidate context',
                    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
                if (folder/'config.json').exists():
                    old=json.loads((folder/'config.json').read_text());signature['source_commit']=old['source_commit']
                    if old['source_sha256']!=signature['source_sha256']:
                        current=signature['source_sha256'];signature['source_sha256']=old['source_sha256'];assert old==signature
                        atomic_json(dict(old_source=old['source_sha256'],new_source=current,reason='Scheduling-only update: both split smokes first and same-runtime formal control after historical parity drift; unchanged intervention policy SHA'),folder/'orchestration_migration.json')
                    assert old==signature
                else:atomic_json(signature,folder/'config.json')
                manifest=[dict(index=int(i),episode_id=list(data[i].episode_id),map_name=data[i].map_name) for i in indices]
                if (folder/'episodes.json').exists():assert json.loads((folder/'episodes.json').read_text())==manifest
                else:atomic_json(manifest,folder/'episodes.json')
                if (folder/'metrics.json').exists():continue
                trace=folder/'predictions.jsonl';done=[json.loads(l) for l in trace.read_text().splitlines()] if trace.exists() else []
                assert [r['episode_id'] for r in done]==[r['episode_id'] for r in manifest[:len(done)]]
                scores=[r['metrics'] for r in done];runtime.control_arm=arm
                with trace.open('a') as f:
                    for pos,i in enumerate(indices):
                        if pos<len(done):continue
                        record=data[i]
                        public=PublicEpisode(record.map_name,record.instruction,record.start_pose.copy(),tuple(catalog.referenced_landmarks(record)),tuple(catalog.referenced_landmark_names(record)))
                        runtime.set_episode(public,builder.transform(public.map_name),catalog)
                        runtime.oracle_goal_xy=record.target_position[:2].copy() if arm.startswith('oracle_') else None
                        poses,term,timings,diag=rollout(public,builder,controller,runtime.predict,ground[public.map_name],protocol,synchronize=torch.cuda.synchronize)
                        score=score_trace(poses,record.target_position,record.trajectory,term,decision_seconds=timings)
                        row=dict(index=int(i),episode_id=list(record.episode_id),map_name=record.map_name,poses=poses.tolist(),termination=term,decision_seconds=timings,step_diagnostics=diag,metrics=score,
                            events=events(poses,diag,record.target_position,term))
                        f.write(json.dumps(row)+'\n');f.flush();scores.append(score)
                        if (pos+1)%25==0:print(json.dumps(dict(arm=arm,split=split,smoke=smoke,completed=pos+1,total=len(indices))),flush=True)
                atomic_json(dict(status='Done',signature=signature,metrics=aggregate(scores)),folder/'metrics.json')
    atomic_json(dict(status='Done',new_conditions=6,archived_baseline_preserved=True,same_runtime_baseline_revalidation=True),OUT/'control_complete.json')

HERE=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True);atomic_json(dict(status='Blocked',error=traceback.format_exc()),OUT/'control_failure.json');raise
