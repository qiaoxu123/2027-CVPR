"""Serial real pilot or complete frozen Seen2470/Unseen2697 evaluation.

There is deliberately no --limit, controller override, threshold override or
benchmark model selection option. Private targets are used after rollout only.
"""
import argparse
from collections import Counter,defaultdict
from dataclasses import asdict
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from .paths import ROOT,DATA,RGBD,CHECKPOINT,A0_SHA,save,sha,bootstrap
from .collected_snapshot import BASE
from .revision_confirmation import immutable_json,CONFIRMATION_CONFIG
from .revision_eval_protocol import (FULL_COUNTS,evaluation_code_hashes,require_population,write_trace,read_trace,
                                     sensor_payload,target_observation_coverage,recall_row,pilot_decision,controller_consistency)
from .revision_freeze import require_candidate,validate_frozen
bootstrap()
from graphnav.closedloop import PublicEpisode,PublicCatalog,ClosedLoopProtocol,rollout,score_trace,aggregate


def pilot_entries():
    population=json.loads((BASE/'revision_v1/confirmation_population.json').read_text())
    entries=[e for e in population['entries'] if e['fold']=='verify']
    entries=sorted(entries,key=lambda e:hashlib.sha256(f"revision-pilot:0:{e['map_name']}:{e['object_id']}".encode()).digest())[:CONFIRMATION_CONFIG['pilot_groups']]
    require_population('pilot',entries);return entries


def source_entries(dataset,scope):
    if scope=='pilot':return pilot_entries()
    entries=[dict(index=r.index,map_name=r.map_name,object_id=r.object_id,description_id=r.description_id,episode_id=list(r.episode_id)) for r in dataset]
    require_population(scope,entries);return entries


def rescore_row(row,record,protocol,builder,catalog):
    from sbfnav.mapdata import GROUND_LEVEL
    if list(row['episode_id'])!=list(record.episode_id) or row['map_name']!=record.map_name:raise ValueError('Stored trace source identity changed')
    if len(row['decision_seconds'])!=len(row['step_diagnostics']) or len(row['observations'])!=len(row['step_diagnostics']):raise ValueError('Incomplete observed decision trace')
    metrics=score_trace(row['poses'],record.target_position,record.trajectory,row['termination'],radius=protocol.success_radius_m,decision_seconds=row['decision_seconds'])
    for k,v in metrics.items():
        if not np.isclose(v,row['metrics'][k],rtol=0,atol=1e-9):raise ValueError(f'Trace metric differs on rescore: {k}')
    if not np.array_equal(np.asarray(row['poses'][0]),record.start_pose):raise ValueError('Trace start differs from source episode')
    transform=builder.transform(record.map_name)
    controller_consistency(row,(transform.x_min,transform.y_min,transform.x_max,transform.y_max),GROUND_LEVEL[record.map_name],protocol)
    if recall_row(row['step_diagnostics'],record.target_position[:2])!=row['recall']:raise ValueError('Recall differs on rescore')
    kind={'Car':0,'Building':1}.get(catalog.target_object(record)['object_type'],-1)
    coverage=[target_observation_coverage(s,record.target_position[:2],kind) for s in row['observations']]
    if coverage!=row['coverage']:raise ValueError('Candidate coverage differs on rescore')
    return metrics


def summarize_rows(rows,dataset):
    maps=defaultdict(list);recalls=[];queries=set();visible=covered=near=updates=steps=caps=errors=0;fallback_episodes=0
    by_index={r.index:r for r in dataset}
    for row in rows:
        record=by_index[row['index']];maps[record.map_name].append(row['metrics']);query=(record.map_name,record.instruction)
        if query not in queries:queries.add(query);recalls.append(row['recall'])
        ds=row['step_diagnostics'];updates+=sum(bool(d.get('a2_updated',False)) for d in ds);steps+=len(ds)
        fallback_episodes+=int(not any(d.get('a2_updated',False) for d in ds))
        caps+=sum(d.get('a2_candidates',0)>=300 for d in ds);errors+=sum(bool(d.get('a2_detection_error')) for d in ds)
        for c in row['coverage']:
            visible+=int(c['target_visible']);covered+=int(c['candidate_point_covered']);near+=int(c['candidate_within20m'])
    return dict(metrics=aggregate([r['metrics'] for r in rows]),by_map={k:aggregate(v) for k,v in maps.items()},
                goal_retrieval=dict(unique_queries=len(recalls),**{k:float(np.mean([r[k] for r in recalls])) for k in recalls[0]}),
                branch=dict(observed_steps=steps,updated_steps=updates,fallback_step_fraction=(steps-updates)/steps if steps else None,
                            fallback_episodes=fallback_episodes,fallback_episode_fraction=fallback_episodes/len(rows),
                            detector_cap_steps=caps,detector_failure_steps=errors),
                candidate_coverage=dict(visible_target_views=visible,point_covered_views=covered,within20m_views=near,
                    point_coverage=covered/visible if visible else None,within20m_coverage=near/visible if visible else None,
                    definition='Post-rollout GT target point inside actual same-category WALDO projected polygon; separate20m polygon buffer; RGB-valid visible denominator'))


def run():
    parser=argparse.ArgumentParser();parser.add_argument('--scope',choices=('pilot','val_seen','val_unseen'),required=True);parser.add_argument('--arm',choices=('A0','A2'),required=True);a=parser.parse_args()
    root=BASE/'revision_v1';checkpoint=root/'training/best.pt';candidate_path=root/'calibration/candidate_calibration.json'
    candidate=json.loads(candidate_path.read_text());require_candidate(candidate)
    if candidate['checkpoint_sha256']!=sha(checkpoint):raise ValueError('Unique calibration checkpoint changed')
    if a.scope=='pilot':
        from .revision_artifacts import verify_revision_training
        verify_revision_training();calibration=candidate_path;freeze_digest=None
    else:
        validate_frozen(root/'frozen.json');calibration=root/'final_calibration.json';freeze_digest=sha(root/'frozen.json')
    import cv2
    from graphnav.dynamic_runtime import NativeObservationBuilder
    from graphnav.stage1_runtime import GraphNavStage1Policy
    from sbfnav.dataset import CityNavDataset,CityReferCatalog
    from sbfnav.planning_state import PlanningStateBuilder
    from sbfnav.navigation import BoundedWaypointController
    from sbfnav.mapdata import GROUND_LEVEL
    from .revision_policy import RevisionPolicy
    torch.set_num_threads(2);torch.manual_seed(0);cv2.setNumThreads(1)
    dataset=CityNavDataset(DATA,'train_seen' if a.scope=='pilot' else a.scope);catalog=CityReferCatalog(DATA/'cityrefer')
    entries=source_entries(dataset,a.scope);require_population(a.scope,entries)
    directory=(root/'pilot'/a.arm) if a.scope=='pilot' else (root/'evaluation'/a.scope/a.arm);directory.mkdir(parents=True,exist_ok=True)
    immutable_json(entries,directory/'episodes.json')
    protocol=ClosedLoopProtocol()
    signature=dict(scope=a.scope,arm=a.arm,checkpoint_sha256=sha(checkpoint),a0_sha256=A0_SHA,calibration_sha256=sha(calibration),
                   freeze_sha256=freeze_digest,code_sha256=evaluation_code_hashes(),episode_manifest_sha256=sha(directory/'episodes.json'),
                   annotation_sha256=sha(dataset.annotation_path),protocol=asdict(protocol),no_target_policy_access=True)
    immutable_json(signature,directory/'signature.json');digest=sha(directory/'signature.json')
    builder=NativeObservationBuilder(PlanningStateBuilder(DATA,PublicCatalog(catalog),canvas_size=384,field_size=28),RGBD,directory/'render_unused')
    policy=GraphNavStage1Policy(CHECKPOINT,directory/'query') if a.arm=='A0' else RevisionPolicy(directory/'query',checkpoint,calibration)
    if policy.args.grid_size!=28 or policy.stop_threshold!=.3 or policy.execution!='guarded-blend' or policy.explore_candidates!=0:
        raise ValueError('Original frozen controller/grid configuration required')
    controller=BoundedWaypointController(horizontal_step_m=50.,vertical_step_m=10.,minimum_above_ground_m=10.,maximum_above_ground_m=100.)
    trace_dir=directory/'traces';trace_dir.mkdir(exist_ok=True);rows=[];started=time.perf_counter()
    for number,entry in enumerate(entries,1):
        record=dataset[entry['index']]
        if (record.map_name,record.object_id,record.description_id)!=(entry['map_name'],entry['object_id'],entry['description_id']):raise ValueError('Live evaluation source identity changed')
        path=trace_dir/f"episode_{record.index:06d}.json"
        if path.exists() and path.with_suffix('.sha.json').exists():
            row=read_trace(path,digest,record.index);rescore_row(row,record,protocol,builder,catalog);rows.append(row);continue
        public=PublicEpisode(record.map_name,record.instruction,record.start_pose.copy(),tuple(catalog.referenced_landmarks(record)),tuple(catalog.referenced_landmark_names(record)))
        policy.set_episode(public,builder.transform(record.map_name),catalog);observations=[]
        def predict(state,instruction):
            prediction=policy.predict(state,instruction)
            candidates=policy.meta['candidates'] if a.arm=='A2' else []
            observations.append(sensor_payload(state.current_observation,candidates))
            prediction.diagnostics=dict(prediction.diagnostics,commanded_xyz=np.asarray(prediction.xyz).tolist(),commanded_stop=bool(prediction.should_stop))
            return prediction
        poses,termination,timings,diagnostics=rollout(public,builder,controller,predict,GROUND_LEVEL[record.map_name],protocol,synchronize=torch.cuda.synchronize)
        # Private fields are first used AFTER the complete closed-loop rollout.
        metrics=score_trace(poses,record.target_position,record.trajectory,termination,radius=protocol.success_radius_m,decision_seconds=timings)
        target_kind={'Car':0,'Building':1}.get(catalog.target_object(record)['object_type'],-1)
        coverage=[target_observation_coverage(s,record.target_position[:2],target_kind) for s in observations]
        row=dict(index=record.index,episode_id=list(record.episode_id),map_name=record.map_name,signature_sha256=digest,
                 poses=poses.tolist(),termination=termination,decision_seconds=timings,step_diagnostics=diagnostics,
                 observations=observations,coverage=coverage,metrics=metrics,recall=recall_row(diagnostics,record.target_position[:2]))
        rescore_row(row,record,protocol,builder,catalog);write_trace(row,path);rows.append(row)
        if number%8==0:
            progress=dict(scope=a.scope,arm=a.arm,completed=number,total=len(entries),seconds=time.perf_counter()-started)
            save(progress,directory/'progress.json');print(json.dumps(progress),flush=True)
    if len(rows)!=len(entries):raise ValueError('Incomplete evaluation rows')
    # Canonical combined trace only after every committed source episode exists.
    with (directory/'predictions.tmp').open('w') as stream:
        for row in rows:stream.write(json.dumps(row)+'\n')
    (directory/'predictions.tmp').replace(directory/'predictions.jsonl')
    report=dict(scope=a.scope,arm=a.arm,signature_sha256=digest,**summarize_rows(rows,dataset),seconds=time.perf_counter()-started,
                checkpoint_sha256=sha(checkpoint),a0_sha256=A0_SHA,calibration_sha256=sha(calibration),code_sha256=evaluation_code_hashes(),
                warning='Shared orthophoto/vertical-depth macro-action benchmark; no perspective-camera or swept3D collision claim',
                recall_note='Unique map+instruction first source target; initial/first/final observed beliefs; final observation may precede final pose',
                latency_note='Synchronized full decision including public trace bookkeeping; one GPU evaluator; resume/cache warmth may affect timing')
    if a.arm=='A0':report.update(branch=None,candidate_coverage=None)
    save(report,directory/'metrics.json')
    save(dict(status='complete',episodes=len(rows),signature_sha256=digest,predictions_sha256=sha(directory/'predictions.jsonl'),metrics_sha256=sha(directory/'metrics.json')),directory/'complete.json')
    print(json.dumps(dict(scope=a.scope,arm=a.arm,metrics=report['metrics'])),flush=True)


def verify_pilot_outputs():
    from sbfnav.dataset import CityNavDataset,CityReferCatalog
    from sbfnav.planning_state import PlanningStateBuilder
    root=BASE/'revision_v1';dataset=CityNavDataset(DATA,'train_seen');entries=pilot_entries();require_population('pilot',entries)
    catalog=CityReferCatalog(DATA/'cityrefer');builder=PlanningStateBuilder(DATA,PublicCatalog(catalog),canvas_size=384,field_size=28)
    arms=[];digests={};code=evaluation_code_hashes();candidate=root/'calibration/candidate_calibration.json'
    for arm in ('A0','A2'):
        directory=root/'pilot'/arm;signature=json.loads((directory/'signature.json').read_text());complete=json.loads((directory/'complete.json').read_text())
        if (signature['scope']!='pilot' or signature['arm']!=arm or signature['code_sha256']!=code
            or signature['calibration_sha256']!=sha(candidate) or signature['checkpoint_sha256']!=sha(root/'training/best.pt')
            or signature['annotation_sha256']!=sha(dataset.annotation_path) or signature['protocol']!=asdict(ClosedLoopProtocol())
            or signature['episode_manifest_sha256']!=sha(directory/'episodes.json') or complete['episodes']!=64 or complete['status']!='complete'
            or complete['signature_sha256']!=sha(directory/'signature.json') or complete['predictions_sha256']!=sha(directory/'predictions.jsonl')
            or complete['metrics_sha256']!=sha(directory/'metrics.json') or json.loads((directory/'episodes.json').read_text())!=entries):
            raise ValueError('Complete paired pilot provenance mismatch')
        digest=sha(directory/'signature.json');rows=[]
        for e in entries:
            row=read_trace(directory/'traces'/f"episode_{e['index']:06d}.json",digest,e['index']);rescore_row(row,dataset[e['index']],ClosedLoopProtocol(),builder,catalog);rows.append(row)
        arms.append(rows);digests[arm]=dict(complete_sha256=sha(directory/'complete.json'),metrics_sha256=sha(directory/'metrics.json'))
    decision=pilot_decision(*arms)
    report=dict(decision=decision,checkpoint_sha256=sha(root/'training/best.pt'),calibration_candidate_sha256=sha(candidate),
                code_sha256=code,arms=digests,a0=aggregate([r['metrics'] for r in arms[0]]),a2=aggregate([r['metrics'] for r in arms[1]]),benchmark_used=False)
    save(report,root/'pilot/report.json');return report


if __name__=='__main__':run()
