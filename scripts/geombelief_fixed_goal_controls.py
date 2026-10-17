"""Strict controller attribution with an identical causal initial belief per episode.

All arms share the original complete model's first-observation field. Subsequent
RGB/belief updates are disabled uniformly, so candidates cannot drift with path.
This is a controlled diagnostic, reported separately from native dynamic policy.
"""
import json,time,os,fcntl,sys,traceback,subprocess
from types import SimpleNamespace
from pathlib import Path
import torch
import numpy as np
from graphnav.checkpointing import atomic_json,atomic_save,file_sha256
from graphnav.full_localization import GoalField
from graphnav.stage1_runtime import GraphNavStage1Policy
from graphnav.a3_runtime import A3TrajectoryPolicy
from graphnav.geombelief_control import ARMS,controlled_predict_field,ORIGINAL_PREDICT_FIELD
from graphnav.closedloop import PublicEpisode,rollout,score_trace,aggregate
from graphnav.dynamic_runtime import NativeObservationBuilder
from graphnav.workspace_paths import WORKSPACE
from scripts.geombelief_map_experiments import OUT,HERE,init
from scripts.polygon_ablation_navigation import resources,PoseOnlyBuilder
from scripts.geombelief_control_experiments import events

KEYS=('belief_logits','candidate_ids','candidate_xy','candidate_context')

def public_episode(record,catalog):
    return PublicEpisode(record.map_name,record.instruction,record.start_pose.copy(),tuple(catalog.referenced_landmarks(record)),tuple(catalog.referenced_landmark_names(record)))

def main():
    while not (OUT/'all_static_complete.json').exists():time.sleep(30)
    plan=init();lock=(OUT/'fixed_goal_run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(1)
    if (OUT/'fixed_goal_complete.json').exists():return
    atomic_json(dict(pid=os.getpid(),started=time.time()),OUT/'fixed_goal_process.json')
    cache=OUT/'initial_belief_cache';cache.mkdir(exist_ok=True)
    runtime=GraphNavStage1Policy(plan['base_checkpoint'],OUT/'fixed_goal_inputs')
    head=runtime.head;captured={}
    def capture(self,state,field):
        captured['field']=field
        return ORIGINAL_PREDICT_FIELD(self,state,field)
    A3TrajectoryPolicy.predict_field=capture
    for split in ('val_seen','val_unseen'):
        data,catalog,base,controller,protocol,ground=resources(plan,split)
        builder=NativeObservationBuilder(base,WORKSPACE/'DATA/sbfnav-data/rgbd',OUT/'fixed_goal_observations'/split)
        for i in range(len(data)):
            path=cache/f'{split}_{i:05d}.pt'
            if path.exists():
                saved=torch.load(path,map_location='cpu',weights_only=True)
                assert saved['episode_id']==list(data[i].episode_id) and saved['checkpoint_sha256']==file_sha256(plan['base_checkpoint'])
                continue
            public=public_episode(data[i],catalog);runtime.set_episode(public,builder.transform(public.map_name),catalog)
            state=builder.build_from_poses(public,np.asarray([public.start_pose]));runtime.predict(state,public.instruction)
            field=captured.pop('field')
            atomic_save(dict(episode_id=list(data[i].episode_id),map_name=public.map_name,start_pose=public.start_pose.tolist(),world_bounds=runtime.record['world_bounds'],instruction=public.instruction,
                checkpoint_sha256=file_sha256(plan['base_checkpoint']),field={k:getattr(field,k).detach().cpu().clone() for k in KEYS},source='Original complete frozen C model; actual initial observation only; no target or teacher trajectory stored'),path)
            if (i+1)%100==0:print(json.dumps(dict(phase='initial_belief_cache',split=split,completed=i+1,total=len(data))),flush=True)
    A3TrajectoryPolicy.predict_field=ORIGINAL_PREDICT_FIELD
    del runtime;torch.cuda.empty_cache()
    atomic_json(dict(status='Done',episodes=5167,checkpoint_sha256=file_sha256(plan['base_checkpoint']),GT_access=False,first_observation_only=True),OUT/'initial_belief_cache_complete.json')
    for smoke in (True,False):
        for split in ('val_seen','val_unseen'):
            data,catalog,base,controller,protocol,ground=resources(plan,split)
            from sbfnav.validation import evenly_spaced_indices
            indices=evenly_spaced_indices(len(data),128 if smoke else None)
            builder=PoseOnlyBuilder(base)
            for arm in ARMS:
                folder=OUT/'controller_fixed_goal'/arm/(split+('_smoke128' if smoke else ''));folder.mkdir(parents=True,exist_ok=True)
                signature=dict(scope='identical_initial_belief',arm=arm,split=split,smoke=smoke,checkpoint_sha256=file_sha256(plan['base_checkpoint']),source_sha256=file_sha256(__file__),policy_sha256=file_sha256(HERE/'graphnav/geombelief_control.py'),protocol=protocol.__dict__,stop_threshold=.3,arrival_guard_m=.001,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),future_observations=False)
                if (folder/'config.json').exists():
                    old=json.loads((folder/'config.json').read_text());signature['source_commit']=old['source_commit'];assert signature==old
                else:atomic_json(signature,folder/'config.json')
                if (folder/'metrics.json').exists():continue
                manifest=[dict(index=int(i),episode_id=list(data[i].episode_id),map_name=data[i].map_name) for i in indices]
                if (folder/'episodes.json').exists():assert json.loads((folder/'episodes.json').read_text())==manifest
                else:atomic_json(manifest,folder/'episodes.json')
                trace=folder/'predictions.jsonl';done=[json.loads(l) for l in trace.read_text().splitlines()] if trace.exists() else []
                assert [r['episode_id'] for r in done]==[r['episode_id'] for r in manifest[:len(done)]]
                scores=[r['metrics'] for r in done]
                with trace.open('a') as f:
                    for position,i in enumerate(indices):
                        if position<len(done):continue
                        record=data[i];public=public_episode(record,catalog);path=cache/f'{split}_{i:05d}.pt';saved=torch.load(path,map_location='cpu',weights_only=True)
                        assert saved['episode_id']==list(record.episode_id) and saved['instruction']==public.instruction and saved['start_pose']==public.start_pose.tolist()
                        field=GoalField(**{k:v.cuda() for k,v in saved['field'].items()})
                        policy=SimpleNamespace(head=head,device=torch.device('cuda'),record=dict(world_bounds=saved['world_bounds'],map_name=public.map_name),execution='guarded-blend',stop_threshold=.3,stop_positive_weight=20.,explore_candidates=0,visited_goal_cells=set(),control_arm=arm,oracle_goal_xy=record.target_position[:2].copy() if arm.startswith('oracle_') else None)
                        def predict(state,instruction):return controlled_predict_field(policy,state,field)
                        poses,term,timings,diag=rollout(public,builder,controller,predict,ground[public.map_name],protocol,synchronize=torch.cuda.synchronize)
                        score=score_trace(poses,record.target_position,record.trajectory,term,decision_seconds=timings)
                        row=dict(index=int(i),episode_id=list(record.episode_id),map_name=record.map_name,poses=poses.tolist(),termination=term,step_diagnostics=diag,decision_seconds=timings,metrics=score,events=events(poses,diag,record.target_position,term),initial_field_sha256=file_sha256(path),belief_frozen_at_initial_observation=True,latency_excludes_initial_field_precomputation=True)
                        f.write(json.dumps(row)+'\n');f.flush();scores.append(score)
                        if (position+1)%200==0:print(json.dumps(dict(phase='fixed_goal_navigation',arm=arm,split=split,smoke=smoke,completed=position+1,total=len(indices))),flush=True)
                atomic_json(dict(status='Done',signature=signature,metrics=aggregate(scores),latency_not_comparable_to_native_full_model=True),folder/'metrics.json')
    atomic_json(dict(status='Done',full_split_evaluations=12,smoke_split_evaluations=12,identical_initial_candidate_fields=True),OUT/'fixed_goal_complete.json')

if __name__=='__main__':
    try:main()
    except Exception:
        atomic_json(dict(status='Blocked',error=traceback.format_exc()),OUT/'fixed_goal_failure.json');raise
