"""New independent targets, frozen-A0 actions, risk head shadowOFF only."""
import argparse,json,time
from collections import Counter
from pathlib import Path
import torch
from .paths import ROOT,DATA,RGBD,save,sha
from .collected_snapshot import BASE
from .revision_confirmation import immutable_json,confirmation_entries
from .calibration_protocol import CALIBRATION_CONFIG
from .direct_risk_protocol import GAIN_QUANTILES,HARM_QUANTILES

CONFIRMATION_CONFIG=dict(seed=0,targets=256,one_query_per_target=True,fixed_action=.25,
    observation_source='real frozen-A0 closed-loop horizon20; direct-risk head shadowOFF',
    temperatures=[1.],gain_quantiles=list(GAIN_QUANTILES),harm_quantiles=list(HARM_QUANTILES),
    quantile_method='numpy linear; FITavailable only; frozen numeric thresholds on VERIFY',
    temperature_target='fixed1, no temperature estimation; corrected class outputs are scores, not probability guarantees',
    class_weight_correction='subtract saved TRAIN logclassweights before temperature; no calibrated-probability guarantee',
    risk_criteria={k:v for k,v in CALIBRATION_CONFIG.items() if (k.startswith(('min_','max_','pilot_')) or k=='bootstrap_replicates') and not k.startswith('min_temperature')},
    no_verify_reselection=True,benchmark_used=False)


def runtime_code_hashes():
    names=('direct_risk_policy.py','direct_risk_runtime.py','direct_risk_head.py','risk_probe.py','revision_policy.py',
           'revision_runtime.py','revision_model.py','revision_fusion.py','policy.py','perception.py','paths.py')
    return {n:sha(ROOT/'experiments/a2_waldo_landmark_belief'/n) for n in names}


def confirmation_code_hashes():
    names=('direct_risk_confirmation.py','direct_risk_train.py','direct_risk_data.py','direct_risk_protocol.py',
           'direct_risk_preflight.py','revision_confirmation.py','revision_eval_protocol.py','calibration_protocol.py')
    return dict(runtime_code_hashes(),**{n:sha(ROOT/'experiments/a2_waldo_landmark_belief'/n) for n in names})


def verify_new_reserve(reserve,previous,training_targets,*,expected=256):
    keys=[(r['map_name'],r['object_id']) for r in reserve['targets']]
    old={(r['map_name'],r['object_id']) for r in previous['targets']}
    if len(keys)!=expected or len(set(keys))!=expected or set(keys)&(old|set(training_targets)):
        raise ValueError('Independent reserve reuses observed/TRAIN targets or is incomplete')


def validate_completed_episode(path,directory,entry,signature,record,bounds,ground,protocol):
    import numpy as np
    from .revision_eval_protocol import controller_consistency
    from graphnav.closedloop import score_trace
    path=Path(path);directory=Path(directory);bound=json.loads(path.with_suffix('.sha.json').read_text())
    if bound!=dict(complete_sha256=sha(path),signature_sha256=signature,index=entry['index']):raise ValueError('Confirmation episode commit changed')
    info=json.loads(path.read_text())
    if info['signature_sha256']!=signature or info['entry']!=entry:raise ValueError('Completed confirmation identity changed')
    if (len(info['rows'])!=len(info['step_diagnostics']) or len(info['rows'])!=len(info['decision_seconds'])
        or [r['step'] for r in info['rows']]!=list(range(len(info['rows']))) or not info['rows']):raise ValueError('Incomplete confirmation decisions')
    if not np.array_equal(np.asarray(info['poses'][0]),record.start_pose):raise ValueError('Confirmation start differs from source')
    controller_consistency(info,bounds,ground,protocol)
    metrics=score_trace(info['poses'],record.target_position,record.trajectory,info['termination'],decision_seconds=info['decision_seconds'])
    if set(metrics)!=set(info['metrics']) or any(not np.isclose(v,info['metrics'][k],atol=1e-9,rtol=0) for k,v in metrics.items()):raise ValueError('Confirmation metrics do not reproduce')
    expected_target=torch.as_tensor(np.asarray(record.target_position)[:2],dtype=torch.float32)
    for item in info['rows']:
        raw=directory/item['filename']
        if Path(item['filename']).name!=item['filename'] or item['episode']!=entry['index'] or item['fold']!=entry['fold'] or sha(raw)!=item['sha256']:
            raise ValueError('Confirmation raw commit/identity changed')
        row=torch.load(raw,map_location='cpu',weights_only=False)
        if (row['signature_sha256']!=signature or row['episode']!=entry['index'] or row['step']!=item['step'] or row['fold']!=entry['fold']
            or row['target_instance']!=[entry['map_name'],entry['object_id']] or not torch.equal(row['labels']['target_xy'],expected_target)):
            raise ValueError('Confirmation raw source label/identity changed')
    return info


def prepare_population():
    root=BASE/'revision_v2';reserve=json.loads((BASE/'revision_v2_confirmation_reserve.json').read_text())
    previous=json.loads((BASE/'revision_confirmation_reserve.json').read_text());source=json.loads((BASE/'source_manifest.json').read_text())
    training={(r['map_name'],r['object_id']) for r in source['entries'] if r['split']=='train'}
    verify_new_reserve(reserve,previous,training)
    if reserve['source_manifest_sha256']!=sha(BASE/'source_manifest.json') or reserve['previous_reserve_sha256']!=sha(BASE/'revision_confirmation_reserve.json'):
        raise ValueError('Reserved source lineage changed')
    entries=confirmation_entries(reserve,source)
    immutable_json(dict(entries=entries,reserve_sha256=sha(BASE/'revision_v2_confirmation_reserve.json'),source_manifest_sha256=sha(BASE/'source_manifest.json')),
                   root/'confirmation_population.json')
    immutable_json(CONFIRMATION_CONFIG,root/'confirmation_protocol.json')
    from .prepare import cache_signature
    immutable_json(cache_signature(),root/'sensor_signature.json')
    return entries,source


def collect():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');a=parser.parse_args()
    entries,source=prepare_population()
    if a.prepare_only:
        print(json.dumps(dict(status='new_population_and_protocol_fixed_before_observation',targets=len(entries),folds=dict(Counter(e['fold'] for e in entries)))),flush=True);return
    root=BASE/'revision_v2';directory=root/'confirmation';directory.mkdir(parents=True,exist_ok=True)
    if (directory/'manifest.json').exists():raise ValueError('New confirmation already complete; calibrate/inspect without rewriting it')
    from .direct_risk_train import verify_completed_training
    verification=verify_completed_training()
    if not verification['checkpoint_eligible']:raise ValueError('Ineligible head cannot consume new confirmation reserve')
    head=root/'head_training/best.pt';rank=BASE/'revision_v1/training/best.pt'
    preflight=json.loads((root/'runtime_preflight.json').read_text())
    if (preflight['status']!='direct_risk_real_TRAIN_runtime_preflight_passed' or not preflight['off_parity_passed'] or not preflight['controller_coupling_passed']
        or preflight['checkpoint_sha256']!=sha(head) or preflight['rank_checkpoint_sha256']!=sha(rank)
        or preflight['runtime_code_sha256']!=runtime_code_hashes() or preflight['preflight_code_sha256']!=sha(ROOT/'experiments/a2_waldo_landmark_belief/direct_risk_preflight.py')):
        raise ValueError('Actual direct-risk CUDA/controller preflight required')
    signature=dict(population_sha256=sha(root/'confirmation_population.json'),protocol_sha256=sha(root/'confirmation_protocol.json'),
        protocol_preobservation_revision_sha256=sha(root/'protocol_preobservation_revision.json'),
        sensor_signature_sha256=sha(root/'sensor_signature.json'),checkpoint_sha256=sha(head),rank_checkpoint_sha256=sha(rank),
        training_verification_sha256=sha(root/'head_training/artifact_verification.json'),preflight_sha256=sha(root/'runtime_preflight.json'),
        code_sha256=confirmation_code_hashes(),observation_source=CONFIRMATION_CONFIG['observation_source'])
    immutable_json(signature,directory/'signature.json');digest=sha(directory/'signature.json')
    from .paths import bootstrap
    bootstrap()
    import cv2,numpy as np
    from .direct_risk_policy import DirectRiskPolicy
    from .train import atomic_checkpoint
    from .revision_eval_protocol import sensor_payload,controller_consistency
    from graphnav.closedloop import PublicEpisode,PublicCatalog,ClosedLoopProtocol,rollout,score_trace
    from graphnav.dynamic_runtime import NativeObservationBuilder
    from sbfnav.dataset import CityNavDataset,CityReferCatalog
    from sbfnav.planning_state import PlanningStateBuilder
    from sbfnav.navigation import BoundedWaypointController
    from sbfnav.mapdata import GROUND_LEVEL
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(0)
    dataset=CityNavDataset(DATA,'train_seen');catalog=CityReferCatalog(DATA/'cityrefer')
    if sha(dataset.annotation_path)!=source['annotation_sha256']:raise ValueError('Live TRAIN annotations changed')
    builder=NativeObservationBuilder(PlanningStateBuilder(DATA,PublicCatalog(catalog),canvas_size=384,field_size=28),RGBD,directory/'render_unused')
    policy=DirectRiskPolicy(directory/'query',rank,head)
    controller=BoundedWaypointController(horizontal_step_m=50.,vertical_step_m=10.,minimum_above_ground_m=10.,maximum_above_ground_m=100.)
    protocol=ClosedLoopProtocol();all_rows=[];episodes=[];started=time.perf_counter()
    for number,entry in enumerate(entries,1):
        complete=directory/f"episode_{entry['index']}_complete.json"
        record=dataset[entry['index']]
        if (record.map_name,record.object_id,record.description_id)!=(entry['map_name'],entry['object_id'],entry['description_id']):raise ValueError('Source episode identity changed')
        transform=builder.transform(record.map_name);bounds=(transform.x_min,transform.y_min,transform.x_max,transform.y_max)
        if complete.exists() and complete.with_suffix('.sha.json').exists():
            info=validate_completed_episode(complete,directory,entry,digest,record,bounds,GROUND_LEVEL[record.map_name],protocol)
            all_rows.extend(info['rows']);episodes.append(info);continue
        public=PublicEpisode(record.map_name,record.instruction,record.start_pose.copy(),tuple(catalog.referenced_landmarks(record)),tuple(catalog.referenced_landmark_names(record)))
        policy.set_episode(public,transform,catalog);local=[]
        def observe(state,instruction):
            prediction=policy.predict(state,instruction)
            if policy.local_diagnostics['a2_updated']:raise ValueError('Confirmation must execute unmodified A0 actions')
            prediction.diagnostics=dict(prediction.diagnostics,commanded_xyz=np.asarray(prediction.xyz).tolist(),commanded_stop=bool(prediction.should_stop))
            fields=('scores','reliable','candidate_match_logits','benefit_logit','gate','category_logits')
            row=dict(public={g:{k:v.detach().cpu().clone() for k,v in values.items()} for g,values in policy.pack.items()},
                prediction={k:policy.branch_output[k][0].detach().cpu().clone() for k in fields},
                risk={k:(v.detach().cpu().clone() if torch.is_tensor(v) else v) for k,v in policy.risk_output.items()},
                observation=sensor_payload(state.current_observation,policy.meta['candidates']),meta=policy.meta,
                episode=entry['index'],step=int(state.step),signature_sha256=digest)
            filename=f"view_{entry['index']}_{state.step}.pt";path=directory/filename
            if path.exists():
                old=torch.load(path,map_location='cpu',weights_only=False)
                if old['signature_sha256']!=digest or not torch.allclose(old['public']['spatial']['base_logits'],row['public']['spatial']['base_logits'],atol=1e-5,rtol=0):raise ValueError('Causal partial A0 replay changed')
            atomic_checkpoint(row,path);local.append((filename,row));return prediction
        poses,termination,timings,diagnostics=rollout(public,builder,controller,observe,GROUND_LEVEL[record.map_name],protocol,synchronize=torch.cuda.synchronize)
        # FIRST target-coordinate consumption is after the ENTIRE rollout.
        controller_consistency(dict(poses=poses,termination=termination,step_diagnostics=diagnostics),
            (transform.x_min,transform.y_min,transform.x_max,transform.y_max),GROUND_LEVEL[record.map_name],protocol)
        row_entries=[]
        for filename,row in local:
            row.update(labels=dict(target_xy=torch.as_tensor(np.asarray(record.target_position)[:2],dtype=torch.float32)),
                       target_instance=[record.map_name,record.object_id],fold=entry['fold'])
            atomic_checkpoint(row,directory/filename)
            row_entries.append(dict(filename=filename,sha256=sha(directory/filename),episode=entry['index'],step=row['step'],fold=entry['fold']))
        metrics=score_trace(poses,record.target_position,record.trajectory,termination,decision_seconds=timings)
        info=dict(entry=entry,signature_sha256=digest,rows=row_entries,poses=poses.tolist(),termination=termination,decision_seconds=timings,step_diagnostics=diagnostics,metrics=metrics)
        save(info,complete);save(dict(complete_sha256=sha(complete),signature_sha256=digest,index=entry['index']),complete.with_suffix('.sha.json'))
        validate_completed_episode(complete,directory,entry,digest,record,bounds,GROUND_LEVEL[record.map_name],protocol)
        all_rows.extend(row_entries);episodes.append(info)
        progress=dict(status='collecting_new_independent_A0_risk_confirmation',completed=number,total=256,views=len(all_rows),seconds=time.perf_counter()-started)
        save(progress,directory/'progress.json');print(json.dumps(progress),flush=True)
    if len(episodes)!=256 or len({(i['entry']['map_name'],i['entry']['object_id']) for i in episodes})!=256:raise ValueError('Incomplete new confirmation population')
    manifest=dict(status='direct_risk_independent_A0_confirmation_complete',signature_sha256=digest,target_groups=256,entries=all_rows,
        episodes=[dict(index=i['entry']['index'],complete_file=f"episode_{i['entry']['index']}_complete.json",sha256=sha(directory/f"episode_{i['entry']['index']}_complete.json"),
                       sidecar_sha256=sha((directory/f"episode_{i['entry']['index']}_complete.json").with_suffix('.sha.json'))) for i in episodes],
        fold_views=dict(Counter(i['fold'] for i in all_rows)),observation_source=CONFIRMATION_CONFIG['observation_source'],seconds=time.perf_counter()-started)
    save(manifest,directory/'manifest.json');print(json.dumps({k:v for k,v in manifest.items() if k not in ('episodes','entries')}),flush=True)


if __name__=='__main__':collect()
