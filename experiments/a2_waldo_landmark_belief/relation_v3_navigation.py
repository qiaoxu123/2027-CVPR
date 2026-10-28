"""Actual frozen-A0/V3 paired64 pilot and complete Seen2470/Unseen2697.

No episode-limit, controller, weight or threshold overrides are exposed. Public
episodes alone reach inference; private scoring starts after the whole rollout.
"""
import argparse,fcntl,json,time,shutil
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from .paths import ROOT,DATA,RGBD,CHECKPOINT,A0_SHA,sha,save,bootstrap
from .collected_snapshot import BASE
from .relation_v3_navigation_admission import load_candidate
from .relation_v3_navigation_freeze import validate_frozen,require_assets
from .relation_v3_navigation_sources import code_hashes,source_entries,check_combined,require_cpu_gates
from .relation_v3_navigation_trace import save_belief,validate_observed_artifacts,latency_summary
from .relation_v3_confirmation_protocol import immutable
from .relation_v3_confirmation import save_native_rgb
from .relation_v3_train import require_idle_gpu
from .revision_eval_protocol import (require_population,write_trace,read_trace,sensor_payload,
    target_observation_coverage,recall_row,pilot_decision)
from .revision_evaluation import rescore_row,summarize_rows
bootstrap()
from graphnav.closedloop import PublicEpisode,PublicCatalog,ClosedLoopProtocol,rollout,score_trace,aggregate


def arm_directory(root,scope,arm):
    if scope not in ('pilot','val_seen','val_unseen') or arm not in ('A0','A2'):raise ValueError('Only fixed A0/A2 navigation populations supported')
    return Path(root)/('pilot' if scope=='pilot' else 'evaluation')/(arm if scope=='pilot' else scope+'/'+arm)


def admit(scope,root):
    root=Path(root)
    if scope=='pilot':
        load_candidate(root)
        return root/'calibration/candidate_calibration.json',None
    if scope not in ('val_seen','val_unseen'):raise ValueError('Unknown navigation scope')
    validate_frozen(root)
    return root/'final_calibration.json',sha(root/'frozen.json')


def signature(scope,arm,dataset,directory,root,calibration,freeze):
    return dict(scope=scope,arm=arm,checkpoint_sha256=sha(root/'training/best.pt'),a0_sha256=A0_SHA,
        calibration_sha256=sha(calibration),freeze_sha256=freeze,code_sha256=code_hashes(),
        episode_manifest_sha256=sha(directory/'episodes.json'),annotation_sha256=sha(dataset.annotation_path),
        navigation_protocol_sha256=sha(root/'navigation_protocol.json'),
        navigation_assets_sha256=sha(root/'navigation_assets.json'),protocol=asdict(ClosedLoopProtocol()),no_target_policy_access=True)


def rescore(row,record,protocol,builder,catalog,directory):
    from sbfnav.mapdata import GROUND_LEVEL
    transform=builder.transform(record.map_name)
    validate_observed_artifacts(row,directory,bounds=(transform.x_min,transform.y_min,transform.x_max,transform.y_max),ground=GROUND_LEVEL[record.map_name])
    return rescore_row(row,record,protocol,builder,catalog)


def run(scope,arm):
    root=BASE/'revision_v3';directory=arm_directory(root,scope,arm)
    # All admission and device-idle checks precede model/device construction.
    require_cpu_gates(root)
    calibration,freeze=admit(scope,root);require_assets(root,create=scope=='pilot')
    import cv2
    from graphnav.dynamic_runtime import NativeObservationBuilder
    from graphnav.stage1_runtime import GraphNavStage1Policy,retrieval_ids
    from sbfnav.dataset import CityNavDataset,CityReferCatalog
    from sbfnav.planning_state import PlanningStateBuilder
    from sbfnav.navigation import BoundedWaypointController
    from sbfnav.mapdata import GROUND_LEVEL
    from .relation_v3_policy import RelationV3Policy
    torch.set_num_threads(2);torch.manual_seed(0);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dataset=CityNavDataset(DATA,'train_seen' if scope=='pilot' else scope);catalog=CityReferCatalog(DATA/'cityrefer')
    entries=source_entries(dataset,scope)
    if (directory/'complete.json').exists():
        builder=PlanningStateBuilder(DATA,PublicCatalog(catalog),canvas_size=384,field_size=28)
        verified_arm(scope,arm,dataset,catalog,builder)
        print(json.dumps(dict(scope=scope,arm=arm,status='retained_verified_complete_navigation',episodes=len(entries))),flush=True);return
    require_idle_gpu()
    directory.mkdir(parents=True,exist_ok=True);immutable(entries,directory/'episodes.json')
    signed=signature(scope,arm,dataset,directory,root,calibration,freeze);immutable(signed,directory/'signature.json');digest=sha(directory/'signature.json')
    builder=NativeObservationBuilder(PlanningStateBuilder(DATA,PublicCatalog(catalog),canvas_size=384,field_size=28),RGBD,directory/'render_unused')
    checkpoint=root/'training/best.pt'
    policy=GraphNavStage1Policy(CHECKPOINT,directory/'query') if arm=='A0' else RelationV3Policy(directory/'query',checkpoint,calibration)
    if policy.args.grid_size!=28 or policy.stop_threshold!=.3 or policy.execution!='guarded-blend' or policy.explore_candidates!=0:
        raise ValueError('Original A0 grid/controller configuration required')
    captured={}
    def capture(module,args,out):
        if captured:raise ValueError('Multiple model forwards in one observed decision')
        captured.update(static_logits=out.static.belief_logits.detach().clone(),belief_logits=out.belief.belief_logits.detach().clone())
    hook=policy.model.register_forward_hook(capture)
    controller=BoundedWaypointController(horizontal_step_m=50.,vertical_step_m=10.,minimum_above_ground_m=10.,maximum_above_ground_m=100.)
    protocol=ClosedLoopProtocol();trace_dir=directory/'traces';trace_dir.mkdir(exist_ok=True);rows=[];started=time.perf_counter()
    try:
        for number,entry in enumerate(entries,1):
            record=dataset[entry['index']]
            if ((record.map_name,record.object_id,record.description_id)!=(entry['map_name'],entry['object_id'],entry['description_id'])
                or list(record.episode_id)!=entry['episode_id']):raise ValueError('Original source query changed')
            path=trace_dir/f'episode_{record.index:06d}.json'
            if path.exists() and path.with_suffix('.sha.json').exists():
                row=read_trace(path,digest,record.index);rescore(row,record,protocol,builder,catalog,directory);rows.append(row);continue
            if path.exists():
                archive=directory/'partial_replay_archive';archive.mkdir(exist_ok=True);shutil.copyfile(path,archive/(path.stem+'_'+sha(path)+'.json'))
            public=PublicEpisode(record.map_name,record.instruction,record.start_pose.copy(),tuple(catalog.referenced_landmarks(record)),tuple(catalog.referenced_landmark_names(record)))
            policy.set_episode(public,builder.transform(record.map_name),catalog);observations=[];inference=[]
            def predict(state,instruction):
                captured.clear();torch.cuda.synchronize();tick=time.perf_counter()
                prediction=policy.predict(state,instruction)
                torch.cuda.synchronize();inference.append(time.perf_counter()-tick)
                if set(captured)!= {'static_logits','belief_logits'}:raise ValueError('Native public beliefs were not captured')
                captured.update(world=policy.world.detach(),static_ids=retrieval_ids(captured['static_logits']),belief_ids=retrieval_ids(captured['belief_logits']))
                sensor=sensor_payload(state.current_observation,policy.meta['candidates'] if arm=='A2' else [])
                sensor.update(native_rgb=save_native_rgb(state.current_observation['bgr'],directory),
                    native_belief=save_belief(captured,directory,index=record.index,step=int(state.step),signature=digest))
                observations.append(sensor)
                prediction.diagnostics=dict(prediction.diagnostics,commanded_xyz=np.asarray(prediction.xyz).tolist(),commanded_stop=bool(prediction.should_stop))
                return prediction
            poses,termination,timings,diagnostics=rollout(public,builder,controller,predict,GROUND_LEVEL[record.map_name],protocol,synchronize=torch.cuda.synchronize)
            # Destination coordinates, category and reference trajectory first
            # enter scoring AFTER the WHOLE actual closed-loop episode.
            metrics=score_trace(poses,record.target_position,record.trajectory,termination,radius=protocol.success_radius_m,decision_seconds=timings)
            kind={'Car':0,'Building':1}.get(catalog.target_object(record)['object_type'],-1)
            row=dict(index=record.index,episode_id=list(record.episode_id),map_name=record.map_name,signature_sha256=digest,
                poses=poses.tolist(),termination=termination,decision_seconds=timings,inference_seconds=inference,
                step_diagnostics=diagnostics,observations=observations,
                coverage=[target_observation_coverage(s,record.target_position[:2],kind) for s in observations],
                metrics=metrics,recall=recall_row(diagnostics,record.target_position[:2]))
            rescore(row,record,protocol,builder,catalog,directory);write_trace(row,path);rows.append(row)
            if number%8==0:
                progress=dict(scope=scope,arm=arm,completed=number,total=len(entries),seconds=time.perf_counter()-started)
                save(progress,directory/'progress.json');print(json.dumps(progress),flush=True)
    finally:hook.remove()
    if len(rows)!=len(entries):raise ValueError('Incomplete navigation population')
    require_assets(root)
    if signature(scope,arm,dataset,directory,root,calibration,freeze)!=signed:raise ValueError('Navigation source changed during evaluation')
    with (directory/'predictions.tmp').open('w') as stream:
        for row in rows:stream.write(json.dumps(row)+'\n')
    (directory/'predictions.tmp').replace(directory/'predictions.jsonl');check_combined(directory/'predictions.jsonl',rows)
    report=dict(scope=scope,arm=arm,signature_sha256=digest,**summarize_rows(rows,dataset),policy_latency=latency_summary(rows),
        checkpoint_sha256=sha(checkpoint),a0_sha256=A0_SHA,calibration_sha256=sha(calibration),code_sha256=code_hashes(),
        warning='Shared orthophoto/vertical-depth macro-action benchmark; no perspective or swept3D collision claim',
        recall_note='Unique map+instruction first source target; initial/first/final observed beliefs; final observation may precede final pose',
        timing_note='metrics.inference_ms includes observation serialization; policy_latency separately excludes PNG/PT/JSON serialization',
        seconds=time.perf_counter()-started)
    if arm=='A0':report.update(branch=None,candidate_coverage=None)
    save(report,directory/'metrics.json')
    save(dict(status='complete',episodes=len(rows),signature_sha256=digest,predictions_sha256=sha(directory/'predictions.jsonl'),metrics_sha256=sha(directory/'metrics.json')),directory/'complete.json')
    print(json.dumps(dict(scope=scope,arm=arm,metrics=report['metrics'])),flush=True)


def verified_arm(scope,arm,dataset,catalog,builder):
    root=BASE/'revision_v3';directory=arm_directory(root,scope,arm);calibration,freeze=admit(scope,root)
    entries=source_entries(dataset,scope);require_population(scope,entries)
    if json.loads((directory/'episodes.json').read_text())!=entries:raise ValueError('Completed episode inventory changed')
    actual=json.loads((directory/'signature.json').read_text())
    if actual!=signature(scope,arm,dataset,directory,root,calibration,freeze):raise ValueError('Completed navigation signature changed')
    digest=sha(directory/'signature.json');complete=json.loads((directory/'complete.json').read_text())
    if (complete!=dict(status='complete',episodes=len(entries),signature_sha256=digest,
        predictions_sha256=sha(directory/'predictions.jsonl'),metrics_sha256=sha(directory/'metrics.json'))):raise ValueError('Complete navigation commit changed')
    rows=[]
    for entry in entries:
        index=entry['index'];row=read_trace(directory/'traces'/f'episode_{index:06d}.json',digest,index)
        record=dataset[index]
        if any(getattr(record,k)!=entry[k] for k in ('map_name','object_id','description_id')) or list(record.episode_id)!=entry['episode_id']:
            raise ValueError('Completed source query changed')
        rescore(row,record,ClosedLoopProtocol(),builder,catalog,directory);rows.append(row)
    check_combined(directory/'predictions.jsonl',rows)
    recomputed=summarize_rows(rows,dataset);stored=json.loads((directory/'metrics.json').read_text())
    for k in ('metrics','by_map','goal_retrieval'):
        if recomputed[k]!=stored[k]:raise ValueError('Navigation aggregate differs from committed observations: '+k)
    if stored['policy_latency']!=latency_summary(rows):raise ValueError('Synchronized policy latency changed')
    if arm=='A2' and any(stored[k]!=recomputed[k] for k in ('branch','candidate_coverage')):raise ValueError('Branch/coverage aggregate changed')
    if stored['code_sha256']!=code_hashes() or stored['signature_sha256']!=digest:raise ValueError('Navigation report provenance changed')
    return rows,stored,dict(complete_sha256=sha(directory/'complete.json'),signature_sha256=digest)


def verify_pilot_outputs():
    from sbfnav.dataset import CityNavDataset,CityReferCatalog
    from sbfnav.planning_state import PlanningStateBuilder
    root=BASE/'revision_v3';load_candidate(root);require_assets(root)
    dataset=CityNavDataset(DATA,'train_seen');catalog=CityReferCatalog(DATA/'cityrefer')
    builder=PlanningStateBuilder(DATA,PublicCatalog(catalog),canvas_size=384,field_size=28)
    arms=[verified_arm('pilot',arm,dataset,catalog,builder) for arm in ('A0','A2')]
    decision=pilot_decision(arms[0][0],arms[1][0])
    report=dict(decision=decision,checkpoint_sha256=sha(root/'training/best.pt'),
        calibration_candidate_sha256=sha(root/'calibration/candidate_calibration.json'),code_sha256=code_hashes(),
        arms={arm:result[2] for arm,result in zip(('A0','A2'),arms)},a0=arms[0][1],a2=arms[1][1],benchmark_used=False)
    immutable(report,root/'pilot/report.json');return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scope',choices=('pilot','val_seen','val_unseen'));parser.add_argument('--arm',choices=('A0','A2'))
    parser.add_argument('--verify-pilot',action='store_true');args=parser.parse_args()
    if args.verify_pilot:
        if args.scope or args.arm:parser.error('Pilot verification takes no scope or arm override')
        print(json.dumps(verify_pilot_outputs()),flush=True);return
    if not args.scope or not args.arm:parser.error('Both --scope and --arm are required')
    with (BASE/'gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(args.scope,args.arm)


if __name__=='__main__':main()
