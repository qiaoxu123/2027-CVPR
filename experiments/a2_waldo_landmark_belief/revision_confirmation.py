"""Reserved target protocol and post-rollout labelled A0-on-policy collection.

This is limited independent confirmation, not restart of full TRAIN extraction.
The trained revision is shadow-only here: original A0 determines every action.
"""
import argparse
from collections import Counter
import hashlib,json,time
from pathlib import Path
import torch
from .paths import ROOT,DATA,RGBD,save,sha
from .collected_snapshot import BASE
from .calibration_protocol import calibration_fold

CONFIRMATION_CONFIG=dict(seed=0,targets=256,one_query_per_target=True,
    observation_source='real frozen-A0 closed-loop horizon20; revision shadow OFF',
    temperatures=[.5,1.,2.,4.],scales=[.1,.25,.5,1.],thresholds=[0.,.2,.35,.5,.65,.8,.9,.95],
    temperature_applies_to='benefit logit only; candidate proposal formula unchanged',
    min_temperature_positive=25,min_temperature_negative=25,min_temperature_positive_groups=5,min_temperature_negative_groups=10,
    min_fit_supported_groups=10,max_fit_broken_group_rate=.02,
    pilot_groups=64,no_verify_reselection=True,benchmark_used=False)


def immutable_json(value,path):
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError(f'Immutable confirmation metadata changed: {path}')
    else:save(value,path)


def confirmation_entries(reserve,source):
    by_index={e['index']:e for e in source['entries']};entries=[];groups=set()
    for target in reserve['targets']:
        key=(target['map_name'],target['object_id'])
        if key in groups:raise ValueError('Duplicate confirmation target')
        groups.add(key);choices=[]
        for index in target['episode_indices']:
            row=by_index[index]
            if (row['map_name'],row['object_id'])!=key or row['split']=='train':raise ValueError('Confirmation identity/partition changed')
            choices.append(row)
        if not choices:raise ValueError('Confirmation target has no source episode')
        selected=min(choices,key=lambda e:hashlib.sha256(f"revision-confirm-query:0:{key[0]}:{key[1]}:{e['index']}".encode()).digest())
        entries.append(dict(selected,fold=calibration_fold(*key)))
    return sorted(entries,key=lambda e:hashlib.sha256(f"revision-confirm-order:0:{e['map_name']}:{e['object_id']}".encode()).digest())


def runtime_code_hashes():
    names=('revision_policy.py','revision_runtime.py','revision_model.py','revision_fusion.py','policy.py','perception.py','paths.py')
    return {n:sha(ROOT/'experiments/a2_waldo_landmark_belief'/n) for n in names}


def confirmation_code_hashes():
    names=('revision_confirmation.py','revision_policy.py','revision_runtime.py','revision_model.py','revision_fusion.py',
           'revision_artifacts.py','calibration_protocol.py','policy.py','perception.py','private_labels.py','paths.py')
    return {n:sha(ROOT/'experiments/a2_waldo_landmark_belief'/n) for n in names}


def collect():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=BASE/'revision_v1/confirmation');a=parser.parse_args()
    from .revision_artifacts import verify_revision_training
    state,verification=verify_revision_training()
    if not state['checkpoint_eligible']:raise ValueError('Revision lacks positive internal net gain; do not consume independent reserve')
    checkpoint=BASE/'revision_v1/training/best.pt'
    preflight=json.loads((BASE/'revision_v1/runtime_preflight.json').read_text())
    if (not preflight['off_parity_passed'] or not preflight['controller_coupling_passed']
        or preflight['checkpoint_sha256']!=sha(checkpoint)
        or preflight['runtime_code_sha256']!=runtime_code_hashes()
        or preflight['preflight_code_sha256']!=sha(ROOT/'experiments/a2_waldo_landmark_belief/revision_preflight.py')):
        raise ValueError('Real frozen revision CUDA/controller preflight is required')
    reserve=json.loads((BASE/'revision_confirmation_reserve.json').read_text());source=json.loads((BASE/'source_manifest.json').read_text())
    if reserve['source_manifest_sha256']!=sha(BASE/'source_manifest.json') or len(reserve['targets'])!=256:
        raise ValueError('Confirmation reserve provenance changed')
    entries=confirmation_entries(reserve,source)
    population=dict(entries=entries,reserve_sha256=sha(BASE/'revision_confirmation_reserve.json'),source_manifest_sha256=sha(BASE/'source_manifest.json'))
    immutable_json(population,BASE/'revision_v1/confirmation_population.json')
    immutable_json(CONFIRMATION_CONFIG,BASE/'revision_v1/confirmation_protocol.json')
    a.output.mkdir(parents=True,exist_ok=True)
    signature=dict(population_sha256=sha(BASE/'revision_v1/confirmation_population.json'),
                   protocol_sha256=sha(BASE/'revision_v1/confirmation_protocol.json'),checkpoint_sha256=sha(checkpoint),
                   training_verification_sha256=sha(BASE/'revision_v1/training/artifact_verification.json'),
                   preflight_sha256=sha(BASE/'revision_v1/runtime_preflight.json'),code_sha256=confirmation_code_hashes(),
                   observation_source=CONFIRMATION_CONFIG['observation_source'])
    immutable_json(signature,a.output/'signature.json');digest=sha(a.output/'signature.json')
    from .paths import bootstrap
    bootstrap()
    import cv2,numpy as np
    from .revision_policy import RevisionPolicy
    from .private_labels import PrivateAssociationIndex,label_view
    from .train import atomic_checkpoint
    from graphnav.closedloop import PublicEpisode,PublicCatalog,ClosedLoopProtocol,rollout,score_trace
    from graphnav.dynamic_runtime import NativeObservationBuilder
    from sbfnav.dataset import CityNavDataset,CityReferCatalog
    from sbfnav.planning_state import PlanningStateBuilder
    from sbfnav.navigation import BoundedWaypointController
    from sbfnav.mapdata import GROUND_LEVEL
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(0)
    ds=CityNavDataset(DATA,'train_seen');catalog=CityReferCatalog(DATA/'cityrefer')
    if sha(ds.annotation_path)!=source['annotation_sha256']:raise ValueError('Live TRAIN annotations changed')
    builder=NativeObservationBuilder(PlanningStateBuilder(DATA,PublicCatalog(catalog),canvas_size=384,field_size=28),RGBD,a.output/'render_unused')
    policy=RevisionPolicy(a.output/'query',checkpoint);index=None
    controller=BoundedWaypointController(horizontal_step_m=50.,vertical_step_m=10.,minimum_above_ground_m=10.,maximum_above_ground_m=100.)
    allowed={(t['map_name'],t['object_id']) for t in reserve['targets']};all_rows=[];episodes=[];counts=Counter();started=time.perf_counter()
    for number,entry in enumerate(entries,1):
        complete=a.output/f"episode_{entry['index']}_complete.json"
        if complete.exists():
            info=json.loads(complete.read_text())
            if info['signature_sha256']!=digest or info['entry']!=entry:raise ValueError('Completed confirmation identity changed')
            for item in info['rows']:
                if sha(a.output/item['filename'])!=item['sha256']:raise ValueError('Completed confirmation row changed')
            all_rows.extend(info['rows']);episodes.append(info);continue
        record=ds[entry['index']]
        if (record.map_name,record.object_id,record.description_id)!=(entry['map_name'],entry['object_id'],entry['description_id']):raise ValueError('Live confirmation episode changed')
        public=PublicEpisode(record.map_name,record.instruction,record.start_pose.copy(),tuple(catalog.referenced_landmarks(record)),tuple(catalog.referenced_landmark_names(record)))
        policy.set_episode(public,builder.transform(record.map_name),catalog);local=[]
        def observe(state,instruction):
            prediction=policy.predict(state,instruction)
            if policy.local_diagnostics['a2_updated']:raise ValueError('Confirmation collection must follow original A0')
            fields=('scores','reliable','candidate_match_logits','benefit_logit','gate','category_logits')
            row=dict(public={g:{k:v.detach().cpu().clone() for k,v in vals.items()} for g,vals in policy.pack.items()},
                     prediction={k:policy.branch_output[k][0].detach().cpu().clone() for k in fields},
                     meta=policy.meta,episode=entry['index'],step=int(state.step),signature_sha256=digest)
            filename=f"view_{entry['index']}_{state.step}.pt";path=a.output/filename
            if path.exists():
                old=torch.load(path,map_location='cpu',weights_only=False)
                if old['signature_sha256']!=digest or not torch.allclose(old['public']['spatial']['base_logits'],row['public']['spatial']['base_logits'],atol=1e-5,rtol=0):
                    raise ValueError('Causal partial confirmation replay changed A0')
            atomic_checkpoint(row,path)
            # Retain only label-free sensor pixels until the FULL rollout ends.
            sensor=dict(footprint_world=np.asarray(state.current_observation['footprint_world']).copy(),bgr=state.current_observation['bgr'].copy())
            local.append((filename,row,sensor));return prediction
        poses,termination,timings,diagnostics=rollout(public,builder,controller,observe,GROUND_LEVEL[record.map_name],ClosedLoopProtocol(),synchronize=torch.cuda.synchronize)
        # All target/GT consumption below is after original-A0 decisions finished.
        if index is None:index=PrivateAssociationIndex(catalog)
        row_entries=[]
        for filename,row,sensor in local:
            labels,private=label_view(record,catalog,index,row['public'],row['meta'],sensor,entry['split'])
            for i,detail in enumerate(private['association']):
                oid=detail.get('object_id');same=(record.map_name,oid) in allowed and calibration_fold(record.map_name,oid)==entry['fold'] if oid is not None else False
                if not same:
                    labels['rank_label_mask'][i]=False;labels['offset_valid'][i]=False
                    private['association'][i]=dict(object_id=None,trusted=False,supervision_allowed=False,reason='outside_confirmation_fold')
            row.update(labels=labels,private=private,target_instance=[record.map_name,record.object_id],fold=entry['fold'])
            atomic_checkpoint(row,a.output/filename)
            row_entries.append(dict(filename=filename,sha256=sha(a.output/filename),episode=entry['index'],step=row['step'],fold=entry['fold']))
        metrics=score_trace(poses,record.target_position,record.trajectory,termination,decision_seconds=timings)
        info=dict(entry=entry,signature_sha256=digest,rows=row_entries,poses=poses.tolist(),termination=termination,decision_seconds=timings,diagnostics=diagnostics,metrics=metrics)
        save(info,complete);all_rows.extend(row_entries);episodes.append(info)
        progress=dict(status='collecting_independent_A0_confirmation',completed=number,total=len(entries),views=len(all_rows),seconds=time.perf_counter()-started)
        save(progress,a.output/'progress.json');print(json.dumps(progress),flush=True)
    if len(episodes)!=256 or len({(e['entry']['map_name'],e['entry']['object_id']) for e in episodes})!=256:raise ValueError('Incomplete reserved population')
    manifest=dict(status='independent_A0_confirmation_complete',signature_sha256=digest,entries=all_rows,
                  episodes=[dict(index=e['entry']['index'],complete_file=f"episode_{e['entry']['index']}_complete.json",sha256=sha(a.output/f"episode_{e['entry']['index']}_complete.json")) for e in episodes],
                  target_groups=256,fold_views=dict(Counter(e['fold'] for e in all_rows)),observation_source=CONFIRMATION_CONFIG['observation_source'],seconds=time.perf_counter()-started)
    save(manifest,a.output/'manifest.json');print(json.dumps({k:v for k,v in manifest.items() if k not in ('entries','episodes')}),flush=True)


if __name__=='__main__':collect()
