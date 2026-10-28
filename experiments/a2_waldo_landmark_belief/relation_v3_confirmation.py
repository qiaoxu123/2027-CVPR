"""Limited163-target independent confirmation; actual A0 actions, V3 shadowOFF."""
import argparse,fcntl,hashlib,json,shutil,time
from collections import Counter
from pathlib import Path
import torch
from .paths import ROOT,DATA,RGBD,sha,save
from .collected_snapshot import BASE,verify_cache_origin
from .relation_v3_confirmation_protocol import prepare,immutable,CONFIG
from .relation_v3_train import code_hashes as training_code_hashes,require_idle_gpu
from .relation_v3_provenance import checked_state
from .relation_v3_runtime import public_network_inputs
from .train import atomic_checkpoint


def collection_code_hashes():
    names=('relation_v3_confirmation.py','relation_v3_confirmation_protocol.py','direct_risk_confirmation.py','revision_eval_protocol.py')
    return dict(training_code_hashes(),**{n:sha(ROOT/'experiments/a2_waldo_landmark_belief'/n) for n in names})


def assert_shadow_OFF(policy):
    if (policy.local_diagnostics['a2_alpha']!=0. or policy.local_diagnostics['a2_updated']
        or not torch.equal(policy.original_field.belief_logits,policy.corrected_field.belief_logits)):
        raise ValueError('Independent collection must execute exact original A0 with V3 shadowOFF')


def capture_public_row(policy,*,episode,step,signature,command):
    assert_shadow_OFF(policy);public_network_inputs(policy.pack,policy.meta)
    if policy.branch_output['offset'].any():raise ValueError('V3 offsets must remain zero')
    return dict(public={g:{k:v.detach().cpu().clone() for k,v in values.items()} for g,values in policy.pack.items()},
        prediction={k:v[0].detach().cpu().clone() for k,v in policy.branch_output.items()},
        risk={k:v[0].detach().cpu().clone() for k,v in policy.risk_output.items()},
        meta=policy.meta,episode=episode,step=step,signature_sha256=signature,command=command)


def commands_match(a,b):
    import numpy as np
    x=np.asarray(a['xyz']);y=np.asarray(b['xyz'])
    if x.shape!=(3,) or y.shape!=(3,) or not np.isfinite(x).all() or not np.isfinite(y).all():return False
    return a['should_stop']==b['should_stop'] and (a['should_stop'] or np.allclose(x,y,atol=1e-4,rtol=0))


def save_native_rgb(bgr,directory):
    import cv2,numpy as np
    if bgr.shape!=(384,384,3) or bgr.dtype!=np.uint8:raise ValueError('Actual native384 BGR frame required')
    digest=hashlib.sha256(np.ascontiguousarray(bgr).tobytes()).hexdigest();folder=directory/'rgb';folder.mkdir(exist_ok=True)
    path=folder/(digest+'.png')
    if path.exists():
        if not np.array_equal(cv2.imread(str(path)),bgr):raise ValueError('Saved observed RGB changed')
    else:
        temp=path.with_suffix('.tmp.png')
        if not cv2.imwrite(str(temp),bgr):raise ValueError('Observed RGB save failed')
        temp.replace(path)
    return dict(filename=str(path.relative_to(directory)),sha256=sha(path),bgr_sha256=digest)


def validate_complete(path,directory,entry,signature,record,bounds,ground,protocol):
    from .direct_risk_confirmation import validate_completed_episode
    info=validate_completed_episode(path,directory,entry,signature,record,bounds,ground,protocol)
    for item in info['rows']:
        row=torch.load(directory/item['filename'],map_location='cpu',weights_only=False)
        image=row['observation']['native_rgb'];name=Path(image['filename'])
        if len(name.parts)!=2 or name.parts[0]!='rgb' or name.name!=image['bgr_sha256']+'.png' or sha(directory/name)!=image['sha256']:
            raise ValueError('Committed actual RGB image changed')
    return info


def archive_partial(path,directory):
    if not path.exists():return
    archive=directory/'partial_replay_archive';archive.mkdir(exist_ok=True)
    target=archive/(path.stem+'_'+sha(path)+path.suffix)
    if not target.exists():shutil.copyfile(path,target)


def collect():
    root=BASE/'revision_v3';directory=root/'confirmation';checkpoint=root/'training/best.pt'
    require_idle_gpu();state=checked_state(checkpoint)
    if not state['checkpoint_eligible']:raise ValueError('No eligible full25 candidate; do not consume fresh confirmation')
    prepare();verify_cache_origin(BASE/'collected_snapshot')
    population=json.loads((root/'confirmation_population.json').read_text());entries=population['fresh_entries']
    if len(entries)!=163:raise ValueError('All163 predeclared fresh targets required')
    from .prepare import cache_signature
    sensor=cache_signature();immutable(sensor,root/'confirmation_sensor_signature.json')
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'manifest.json').exists():raise ValueError('Complete confirmation retained; validate/calibrate instead of recollecting')
    signature=dict(population_sha256=sha(root/'confirmation_population.json'),protocol_sha256=sha(root/'confirmation_protocol.json'),
        roles_sha256=state['roles_sha256'],sealed_verify_sha256=sha(root/'sealed_verify130.json'),
        sensor_signature_sha256=sha(root/'confirmation_sensor_signature.json'),checkpoint_sha256=sha(checkpoint),
        training_verification_sha256=sha(root/'training/artifact_verification.json'),code_sha256=collection_code_hashes(),
        observation_source=CONFIG['source'])
    immutable(signature,directory/'signature.json');digest=sha(directory/'signature.json')
    from .paths import bootstrap
    bootstrap()
    import cv2,numpy as np
    from .relation_v3_policy import RelationV3Policy
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
    require_idle_gpu();policy=RelationV3Policy(directory/'query',checkpoint)
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
    manifest=dict(status='v3_fresh163_independent_A0_confirmation_complete',signature_sha256=digest,target_groups=163,entries=all_rows,
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
