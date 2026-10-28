"""Sequential real-CUDA TRAIN teacher replay: A0, exact OFF, controller coupling.

The ON arm is an explicit TRAIN-only functional diagnostic, not calibration,
benchmark evaluation or authorization to deploy the gate.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from .paths import ROOT,DATA,RGBD,CHECKPOINT,save,sha
from .collected_snapshot import BASE,cache_source
from .revision_confirmation import immutable_json,runtime_code_hashes


def check_preflight_records(a0,off,on):
    if not a0 or len(a0)!=len(off) or len(a0)!=len(on):raise ValueError('Incomplete preflight row population')
    fields=goals=actions=0
    for a,b,c in zip(a0,off,on):
        if any((r['index'],r['step'])!=(a['index'],a['step']) for r in (b,c)):raise ValueError('Preflight identities/order changed')
        if (a['should_stop']!=b['should_stop'] or not np.array_equal(a['xyz'],b['xyz'])
            or not np.array_equal(a['belief_top16_world_xy'],b['belief_top16_world_xy'])):
            raise ValueError('Revision OFF changed frozen A0 decisions')
        fields+=int(c['a2_updated'])
        goal_a=a.get('selected_goal_xy',a['belief_top16_world_xy'][0]);goal_c=c.get('selected_goal_xy',c['belief_top16_world_xy'][0])
        goals+=int(not np.array_equal(goal_a,goal_c))
        actions+=int(a['should_stop']!=c['should_stop'] or np.linalg.norm(np.asarray(a['xyz'])-np.asarray(c['xyz']))>1e-4)
    if fields==0 or goals==0 or actions==0:raise ValueError('No nonzero field-to-original-controller coupling exercised')
    return dict(off_parity_passed=True,controller_coupling_passed=True,views=len(a0),changed_fields=fields,changed_selected_goals=goals,changed_actions=actions)


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=('a0','off','on','verify'),required=True);a=p.parse_args()
    root=BASE/'revision_v1';directory=root/'preflight';directory.mkdir(parents=True,exist_ok=True)
    checkpoint=root/'training/best.pt'
    if a.arm!='verify':
        from .revision_artifacts import verify_revision_training
        state,report=verify_revision_training()
        if not state['checkpoint_eligible']:raise ValueError('No positive internal revision gain; do not proceed')
    else:
        marker=json.loads((root/'training/training_complete.json').read_text())
        if marker['rank_epochs']!=20 or marker['refine_epochs']!=5 or marker['checkpoint_sha256']!=sha(checkpoint):raise ValueError('Incomplete final revision checkpoint')
    signature=dict(checkpoint_sha256=sha(checkpoint),runtime_code_sha256=runtime_code_hashes(),preflight_code_sha256=sha(Path(__file__)))
    immutable_json(signature,directory/'signature.json');digest=sha(directory/'signature.json')
    if a.arm=='verify':
        rows=[]
        for arm in ('a0','off','on'):
            data=json.loads((directory/f'{arm}.json').read_text())
            if data['signature_sha256']!=digest:raise ValueError('Preflight arm provenance changed')
            rows.append(data['rows'])
        result=check_preflight_records(*rows)
        save(dict(result,**signature,status='revision_real_TRAIN_runtime_preflight_passed',no_benchmark_data=True,scope='Causal sparse teacher replay, not navigation benefit'),root/'runtime_preflight.json')
        print(json.dumps(result),flush=True);return
    if (directory/f'{a.arm}.json').exists():
        data=json.loads((directory/f'{a.arm}.json').read_text())
        if data['signature_sha256']!=digest:raise ValueError('Existing preflight arm differs')
        print(f'Existing {a.arm} arm retained',flush=True);return
    source=cache_source(BASE/'collected_snapshot');by_index={e['index']:e for e in source['entries']}
    manifest=json.loads((BASE/'collected_snapshot/manifest.json').read_text());groups={}
    for row in manifest['entries']:
        if row['split']!='train' or not row['counts'].get('reference_eligible_views') or not row['counts'].get('candidates'):continue
        entry=by_index[row['episode']];groups.setdefault((entry['map_name'],entry['object_id']),entry)
    selected=sorted(groups.values(),key=lambda e:hashlib.sha256(f"revision-functional:0:{e['map_name']}:{e['object_id']}".encode()).digest())[:32]
    if len(selected)!=32:raise ValueError('Insufficient distinct TRAIN functional samples')
    immutable_json(selected,directory/'episodes.json')
    from .paths import bootstrap
    bootstrap()
    import cv2
    from graphnav.closedloop import PublicEpisode,PublicCatalog
    from graphnav.dynamic_runtime import NativeObservationBuilder
    from graphnav.stage1_runtime import GraphNavStage1Policy
    from sbfnav.dataset import CityNavDataset,CityReferCatalog
    from sbfnav.planning_state import PlanningStateBuilder
    from sbfnav.mapdata import GROUND_LEVEL
    from .revision_policy import RevisionPolicy
    torch.set_num_threads(2);torch.manual_seed(0);cv2.setNumThreads(1)
    ds=CityNavDataset(DATA,'train_seen');catalog=CityReferCatalog(DATA/'cityrefer')
    builder=NativeObservationBuilder(PlanningStateBuilder(DATA,PublicCatalog(catalog),canvas_size=384,field_size=28),RGBD,directory/'render_unused')
    if a.arm=='a0':policy=GraphNavStage1Policy(CHECKPOINT,directory/'query_a0')
    else:
        calibration=None
        if a.arm=='on':
            calibration=directory/'TRAIN_only_uncalibrated_on.json'
            immutable_json(dict(enabled=True,scale=1.,threshold=0.,temperature=1.,checkpoint_sha256=sha(checkpoint),gate_final=False,scope='TRAIN functional intervention only'),calibration)
        policy=RevisionPolicy(directory/f'query_{a.arm}',checkpoint,calibration)
    results=[]
    for number,entry in enumerate(selected,1):
        record=ds[entry['index']]
        public=PublicEpisode(record.map_name,record.instruction,record.start_pose.copy(),tuple(catalog.referenced_landmarks(record)),tuple(catalog.referenced_landmark_names(record)))
        policy.set_episode(public,builder.transform(record.map_name),catalog)
        trajectory=np.asarray(record.trajectory);legal=np.flatnonzero(np.isfinite(trajectory).all(-1)&(trajectory[:,2]-GROUND_LEVEL[record.map_name]>=10)&(trajectory[:,2]-GROUND_LEVEL[record.map_name]<=100))
        steps=sorted({int(legal[round(q*(len(legal)-1))]) for q in (.25,.75,1.)}) if len(legal) else []
        for step in steps:
            observation=builder.build_from_poses(public,trajectory[:step+1]);pred=policy.predict(observation,record.instruction);d=pred.diagnostics
            results.append(dict(index=entry['index'],step=step,xyz=np.asarray(pred.xyz).tolist(),should_stop=bool(pred.should_stop),
                                selected_goal_xy=d['selected_goal_xy'],belief_top16_world_xy=d['belief_top16_world_xy'],a2_updated=d.get('a2_updated',False)))
        print(json.dumps(dict(arm=a.arm,episodes_done=number,views=len(results))),flush=True)
    save(dict(signature_sha256=digest,arm=a.arm,rows=results),directory/f'{a.arm}.json')


if __name__=='__main__':main()
