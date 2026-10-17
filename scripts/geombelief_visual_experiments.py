"""Extra frozen residual-input controls; existing four arms are never rerun."""
import json,sys,os,time,fcntl,traceback,subprocess
from pathlib import Path
import numpy as np
import torch
from graphnav.stage1_runtime import GraphNavStage1Policy
from graphnav.calibrated_residual_runtime import install_calibrated_residual
from graphnav.geombelief_visual_inputs import install_input_intervention
from graphnav.closedloop import PublicEpisode,rollout,score_trace,aggregate
from graphnav.dynamic_runtime import NativeObservationBuilder
from graphnav.checkpointing import file_sha256,atomic_json
from graphnav.workspace_paths import WORKSPACE
from scripts.polygon_ablation_navigation import resources
from scripts.geombelief_map_experiments import OUT,HERE,init

ARMS=('geometry_pose','rgb_text','current_rgb')

def main():
    plan=init();lock=(OUT/'visual_run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(1)
    atomic_json(dict(pid=os.getpid(),started=time.time(),status='Running',training=False),OUT/'visual_process.json')
    cp=HERE/'calibrated_residual/best.pt'
    frozen=json.loads((OUT/'frozen_visual_input_plan.json').read_text())
    assert frozen['residual_sha256']==file_sha256(cp) and frozen['base_checkpoint_sha256']==file_sha256(plan['base_checkpoint'])
    assert frozen['input_source_sha256']==file_sha256(HERE/'graphnav/geombelief_visual_inputs.py')
    for arm in ARMS:
        policy=GraphNavStage1Policy(plan['base_checkpoint'],OUT/'visual_inputs'/arm)
        install_calibrated_residual(policy,cp,plan['base_checkpoint']);install_input_intervention(policy,arm)
        for split in ('val_seen','val_unseen'):
            data,catalog,base,controller,protocol,ground=resources(plan,split)
            folder=OUT/'visual'/arm/split;folder.mkdir(parents=True,exist_ok=True)
            signature=dict(arm=arm,kind='frozen_input_intervention_not_fair_retraining',split=split,
                checkpoint_sha256=file_sha256(plan['base_checkpoint']),residual_sha256=file_sha256(cp),
                source_sha256=file_sha256(__file__),frozen_visual_plan_sha256=file_sha256(OUT/'frozen_visual_input_plan.json'),input_mask_sha256=file_sha256(HERE/'graphnav/geombelief_visual_inputs.py'),
                protocol=protocol.__dict__,stop_threshold=.3,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
            if (folder/'config.json').exists():
                old=json.loads((folder/'config.json').read_text());signature['source_commit']=old['source_commit'];assert signature==old
            else:atomic_json(signature,folder/'config.json')
            if (folder/'metrics.json').exists():continue
            manifest=[dict(index=i,episode_id=list(data[i].episode_id),map_name=data[i].map_name) for i in range(len(data))]
            if (folder/'episodes.json').exists():assert json.loads((folder/'episodes.json').read_text())==manifest
            else:atomic_json(manifest,folder/'episodes.json')
            trace=folder/'predictions.jsonl';done=[json.loads(l) for l in trace.read_text().splitlines()] if trace.exists() else []
            assert [r['episode_id'] for r in done]==[r['episode_id'] for r in manifest[:len(done)]]
            scores=[r['metrics'] for r in done]
            builder=NativeObservationBuilder(base,WORKSPACE/'DATA/sbfnav-data/rgbd',OUT/'visual_observations'/split)
            with trace.open('a') as f:
                for i in range(len(done),len(data)):
                    record=data[i];public=PublicEpisode(record.map_name,record.instruction,record.start_pose.copy(),tuple(catalog.referenced_landmarks(record)),tuple(catalog.referenced_landmark_names(record)))
                    policy.set_episode(public,builder.transform(public.map_name),catalog)
                    poses,term,timings,diag=rollout(public,builder,controller,policy.predict,ground[public.map_name],protocol,synchronize=torch.cuda.synchronize)
                    score=score_trace(poses,record.target_position,record.trajectory,term,decision_seconds=timings)
                    f.write(json.dumps(dict(index=i,episode_id=list(record.episode_id),map_name=record.map_name,poses=poses.tolist(),termination=term,step_diagnostics=diag,decision_seconds=timings,metrics=score))+'\n');f.flush();scores.append(score)
                    if (i+1)%100==0:print(json.dumps(dict(arm=arm,split=split,completed=i+1,total=len(data))),flush=True)
            atomic_json(dict(status='Done',signature=signature,metrics=aggregate(scores)),folder/'metrics.json')
        del policy;torch.cuda.empty_cache()
    atomic_json(dict(status='Done',kind='frozen_input_intervention_not_fair_retraining',full_splits=6),OUT/'visual_complete.json')

if __name__=='__main__':
    try:main()
    except Exception:
        atomic_json(dict(status='Blocked',error=traceback.format_exc()),OUT/'visual_failure.json');raise
