"""V4 independent163-target collection with exact A0 actions and V4 shadowOFF."""
import argparse
import fcntl
import json
import time
from collections import Counter
from pathlib import Path

import torch

from .paths import ROOT,DATA,RGBD,sha,save
from .collected_snapshot import BASE,verify_cache_origin
from .reference_v4_confirmation_protocol import prepare,immutable,CONFIG,BINDING_KEYS
from .reference_v4_train import code_hashes as training_code_hashes
from .reference_v4_provenance import checked_state
from .relation_v3_train import require_idle_gpu
from .relation_v3_confirmation import (
    capture_public_row,commands_match,save_native_rgb,validate_complete,archive_partial,
)
from .train import atomic_checkpoint


def collection_code_hashes():
    names=('reference_v4_confirmation.py','reference_v4_confirmation_protocol.py','reference_v4_confirmation_preparation.py',
           'relation_v3_confirmation.py','direct_risk_confirmation.py','revision_eval_protocol.py',
           'relation_v3_confirmation_protocol.py')
    return dict(training_code_hashes(),**{name:sha(ROOT/'experiments/a2_waldo_landmark_belief'/name) for name in names})


def collection_signature(root,state,checkpoint):
    """Bind an already admitted complete V4 model to its independent inventory."""
    root=Path(root)
    if (state.get('checkpoint_eligible') is not True or state.get('epoch') not in range(21,26)
        or state.get('scope')=='TRAIN_functional_only'
        or state.get('config',{}).get('version')!='a2_relation_v4_public_roads'):
        raise ValueError('Eligible complete V4 model required for independent collection')
    population=json.loads((root/'confirmation_population.json').read_text())
    if any(state.get(key)!=population.get(key) or key not in population for key in BINDING_KEYS):
        raise ValueError('Independent V4 collection belongs to another feature/model lineage')
    if json.loads((root/'confirmation_protocol.json').read_text())!=CONFIG:
        raise ValueError('Independent V4 observation protocol changed')
    return dict(population_sha256=sha(root/'confirmation_population.json'),protocol_sha256=sha(root/'confirmation_protocol.json'),
        roles_sha256=state['roles_sha256'],sealed_verify_sha256=population['sealed_verify_sha256'],
        sensor_signature_sha256=sha(root/'confirmation_sensor_signature.json'),checkpoint_sha256=sha(checkpoint),
        training_verification_sha256=sha(root/'training/artifact_verification.json'),code_sha256=collection_code_hashes(),
        feature_manifest_sha256=state['feature_manifest_sha256'],feature_signature_sha256=state['feature_signature_sha256'],
        feature_build_status_sha256=state['feature_build_status_sha256'],observation_source=CONFIG['source'])


def collect():
    root=BASE/'revision_v4';directory=root/'confirmation';checkpoint=root/'training/best.pt'
    require_idle_gpu();state=checked_state(checkpoint)
    if not state['checkpoint_eligible']:raise ValueError('No eligible full25 candidate; do not consume fresh confirmation')
    prepare()
    from .reference_v4_confirmation_preparation import require_cpu_gates
    require_cpu_gates(root)
    verify_cache_origin(BASE/'collected_snapshot')
    population=json.loads((root/'confirmation_population.json').read_text());entries=population['fresh_entries']
    if len(entries)!=163:raise ValueError('All163 predeclared fresh targets required')
    from .prepare import cache_signature
    sensor=cache_signature();immutable(sensor,root/'confirmation_sensor_signature.json')
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'manifest.json').exists():raise ValueError('Complete confirmation retained; validate/calibrate instead of recollecting')
    signature=collection_signature(root,state,checkpoint)
    immutable(signature,directory/'signature.json');digest=sha(directory/'signature.json')
    from .paths import bootstrap
    bootstrap()
    import cv2,numpy as np
    from .reference_v4_policy import ReferenceV4Policy
    from .revision_eval_protocol import sensor_payload,controller_consistency
    from graphnav.closedloop import PublicEpisode,PublicCatalog,ClosedLoopProtocol,rollout,score_trace
    from graphnav.dynamic_runtime import NativeObservationBuilder
    from sbfnav.dataset import CityNavDataset,CityReferCatalog
    from sbfnav.planning_state import PlanningStateBuilder
    from sbfnav.navigation import BoundedWaypointController
    from sbfnav.mapdata import GROUND_LEVEL
    torch.set_num_threads(2);torch.manual_seed(0);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dataset=CityNavDataset(DATA,'train_seen');catalog=CityReferCatalog(DATA/'cityrefer')
    source=json.loads((BASE/'source_manifest.json').read_text())
    if sha(dataset.annotation_path)!=source['annotation_sha256']:raise ValueError('Live TRAIN annotations changed')
    builder=NativeObservationBuilder(PlanningStateBuilder(DATA,PublicCatalog(catalog),canvas_size=384,field_size=28),RGBD,directory/'render_unused')
    require_idle_gpu();policy=ReferenceV4Policy(directory/'query',checkpoint)
    controller=BoundedWaypointController(horizontal_step_m=50.,vertical_step_m=10.,minimum_above_ground_m=10.,maximum_above_ground_m=100.)
    protocol=ClosedLoopProtocol();all_rows=[];episodes=[];start=time.perf_counter()
    for number,entry in enumerate(entries,1):
        require_idle_gpu(allow_self=True);complete=directory/f"episode_{entry['index']}_complete.json";record=dataset[entry['index']]
        if (record.map_name,record.object_id,record.description_id)!=(entry['map_name'],entry['object_id'],entry['description_id']):raise ValueError('Frozen TRAIN query identity changed')
        transform=builder.transform(record.map_name);bounds=(transform.x_min,transform.y_min,transform.x_max,transform.y_max)
        if complete.exists() and complete.with_suffix('.sha.json').exists():
            info=validate_complete(complete,directory,entry,digest,record,bounds,GROUND_LEVEL[record.map_name],protocol)
            all_rows.extend(info['rows']);episodes.append(info);continue
        archive_partial(complete,directory)
        public=PublicEpisode(record.map_name,record.instruction,record.start_pose.copy(),tuple(catalog.referenced_landmarks(record)),tuple(catalog.referenced_landmark_names(record)))
        policy.set_episode(public,transform,catalog);local=[]
        def observe(observation,instruction):
            prediction=policy.predict(observation,instruction)
            command=dict(xyz=np.asarray(prediction.xyz).tolist(),should_stop=bool(prediction.should_stop))
            prediction.diagnostics=dict(prediction.diagnostics,commanded_xyz=command['xyz'],commanded_stop=command['should_stop'])
            row=capture_public_row(policy,episode=entry['index'],step=int(observation.step),signature=digest,command=command)
            row['observation']=sensor_payload(observation.current_observation,policy.meta['candidates'])
            row['observation']['native_rgb']=save_native_rgb(observation.current_observation['bgr'],directory)
            filename=f"view_{entry['index']}_{observation.step}.pt";path=directory/filename
            if path.exists():
                previous=torch.load(path,map_location='cpu',weights_only=False)
                if (previous['signature_sha256']!=digest or previous['episode']!=entry['index'] or previous['step']!=int(observation.step)
                    or not commands_match(previous['command'],command)
                    or not torch.allclose(previous['public']['spatial']['base_logits'],row['public']['spatial']['base_logits'],atol=1e-5,rtol=1e-5)):
                    raise ValueError('Causal partial A0 replay changed')
                archive_partial(path,directory)
            atomic_checkpoint(row,path);local.append((filename,row));return prediction
        poses,termination,timings,diagnostics=rollout(public,builder,controller,observe,GROUND_LEVEL[record.map_name],protocol,synchronize=torch.cuda.synchronize)
        controller_consistency(dict(poses=poses,termination=termination,step_diagnostics=diagnostics),bounds,GROUND_LEVEL[record.map_name],protocol)
        # Destination coordinates FIRST enter this workflow only now, after
        # the WHOLE actual-A0 rollout. They never enter public network inputs.
        row_entries=[]
        for filename,row in local:
            row.update(labels=dict(target_xy=torch.as_tensor(np.asarray(record.target_position)[:2],dtype=torch.float32)),
                target_instance=[record.map_name,record.object_id],fold=entry['fold'])
            atomic_checkpoint(row,directory/filename)
            row_entries.append(dict(filename=filename,sha256=sha(directory/filename),episode=entry['index'],step=row['step'],fold=entry['fold']))
        metrics=score_trace(poses,record.target_position,record.trajectory,termination,decision_seconds=timings)
        info=dict(entry=entry,signature_sha256=digest,rows=row_entries,poses=poses.tolist(),termination=termination,
            decision_seconds=timings,step_diagnostics=diagnostics,metrics=metrics)
        save(info,complete);save(dict(complete_sha256=sha(complete),signature_sha256=digest,index=entry['index']),complete.with_suffix('.sha.json'))
        validate_complete(complete,directory,entry,digest,record,bounds,GROUND_LEVEL[record.map_name],protocol)
        all_rows.extend(row_entries);episodes.append(info)
        progress=dict(status='collecting_fresh163_A0_shadowOFF_only',completed=number,total=163,views=len(all_rows),seconds=time.perf_counter()-start)
        save(progress,directory/'progress.json');print(json.dumps(progress),flush=True)
    if len(episodes)!=163 or {i['entry']['index'] for i in episodes}!={e['index'] for e in entries}:raise ValueError('Incomplete independent163 population')
    manifest=dict(status='v4_fresh163_independent_A0_confirmation_complete',signature_sha256=digest,target_groups=163,entries=all_rows,
        episodes=[dict(index=i['entry']['index'],complete_file=f"episode_{i['entry']['index']}_complete.json",sha256=sha(directory/f"episode_{i['entry']['index']}_complete.json"),
            sidecar_sha256=sha((directory/f"episode_{i['entry']['index']}_complete.json").with_suffix('.sha.json'))) for i in episodes],
        fold_views=dict(Counter(i['fold'] for i in all_rows)),seconds=time.perf_counter()-start,
        timing_scope='Recorded A0 shadow collection includes serialization/RGB capture; NOT final inference latency')
    save(manifest,directory/'manifest.json');print(json.dumps({k:v for k,v in manifest.items() if k not in ('episodes','entries')}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    if args.prepare_only:prepare();return
    with (BASE/'gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);collect()


if __name__=='__main__':main()
