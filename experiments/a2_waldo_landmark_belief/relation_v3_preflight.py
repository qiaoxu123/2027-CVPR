"""Serial real-TRAIN A0/OFF/ON runtime checks before any V3 full training."""
import argparse,fcntl,hashlib,json
import numpy as np
from .paths import DATA,RGBD,CHECKPOINT,sha,save
from .collected_snapshot import BASE,cache_source,verify_cache_origin
from .relation_v3_train import code_hashes,require_idle_gpu,validate_audit
from .revision_confirmation import immutable_json


def check_records(a0,off,on):
    if not a0 or len(a0)!=len(off) or len(a0)!=len(on):raise ValueError('Incomplete functional arm population')
    fields=goals=actions=0
    for a,b,c in zip(a0,off,on):
        if any((r['index'],r['step'])!=(a['index'],a['step']) for r in (b,c)):raise ValueError('Functional arm identities changed')
        if (a['should_stop']!=b['should_stop'] or not np.array_equal(a['xyz'],b['xyz'])
            or not np.array_equal(a['belief_top16_world_xy'],b['belief_top16_world_xy'])
            or not np.array_equal(a['selected_goal_xy'],b['selected_goal_xy'])
            or any(a[k]!=b[k] for k in ('belief_logits_sha256','belief_logits_shape','belief_logits_dtype'))):
            raise ValueError('OFF changed original A0/controller')
        fields+=int(c['a2_updated']);goals+=int(not np.array_equal(a['selected_goal_xy'],c['selected_goal_xy']))
        changed_stop=a['should_stop']!=c['should_stop']
        changed_motion=not a['should_stop'] and not c['should_stop'] and np.linalg.norm(np.asarray(a['xyz'])-np.asarray(c['xyz']))>1e-4
        actions+=int(changed_stop or changed_motion)
    if not fields or not goals or not actions:raise ValueError('No effective field-to-controller coupling exercised')
    return dict(off_parity_passed=True,controller_coupling_passed=True,views=len(a0),changed_fields=fields,changed_selected_goals=goals,changed_actions=actions)


def prepare_queries(directory,ds,catalog):
    source=cache_source(BASE/'collected_snapshot');by_index={e['index']:e for e in source['entries']}
    manifest=json.loads((BASE/'collected_snapshot/manifest.json').read_text());roles=json.loads((BASE/'revision_v3/roles.json').read_text())
    allowed={(e['map_name'],e['object_id']) for e in roles['target_roles']['train']};groups={}
    for row in manifest['entries']:
        if row['split']!='train' or not row['counts'].get('reference_eligible_views') or not row['counts'].get('candidates'):continue
        e=by_index[row['episode']];key=(e['map_name'],e['object_id'])
        if key in allowed:groups.setdefault(key,e)
    selected=sorted(groups.values(),key=lambda e:hashlib.sha256(f"relation-v3-functional:0:{e['map_name']}:{e['object_id']}".encode()).digest())[:32]
    if len(selected)!=32:raise ValueError('All32 distinct TRAIN functional targets required')
    public=[]
    for e in selected:
        record=ds[e['index']]
        if record.map_name!=e['map_name']:raise ValueError('Original TRAIN query source changed')
        public.append(dict(map_name=record.map_name,instruction=record.instruction,start_pose=record.start_pose.tolist()))
    immutable_json(selected,directory/'episodes.json');immutable_json(public,directory/'public_episodes.json')
    return selected


def run_arm(arm):
    root=BASE/'revision_v3';directory=root/'preflight';directory.mkdir(parents=True,exist_ok=True)
    hashes=code_hashes();roles=sha(root/'roles.json');validate_audit();verify_cache_origin(BASE/'collected_snapshot')
    initial=root/'pretraining/initial.pt';cpu=json.loads((root/'cpu_gates.json').read_text())
    if cpu['code_sha256']!=hashes or cpu['roles_sha256']!=roles or cpu['initial_checkpoint_sha256']!=sha(initial):raise ValueError('Current CPU/initial proof required')
    if arm=='verify':
        signature=json.loads((directory/'signature.json').read_text())
        if (signature['code_sha256']!=hashes or signature['roles_sha256']!=roles
            or signature['initial_checkpoint_sha256']!=sha(initial) or signature['cpu_gates_sha256']!=sha(root/'cpu_gates.json')
            or signature['public_episodes_sha256']!=sha(directory/'public_episodes.json')
            or signature['selected_episodes_sha256']!=sha(directory/'episodes.json')):raise ValueError('Functional signature changed')
        records=[]
        for name in ('a0','off','on'):
            data=json.loads((directory/f'{name}.json').read_text())
            if data['signature_sha256']!=sha(directory/'signature.json') or data['episodes_completed']!=32:raise ValueError('Incomplete functional arm')
            records.append(data['rows'])
        result=check_records(*records)
        save(dict(result,**signature,status='v3_real_TRAIN_runtime_preflight_passed',no_benchmark_used=True,
                  functional_untrained_only=True,scope='Causal sparse teacher replay, not navigation gain'),root/'runtime_preflight.json')
        print(json.dumps(result),flush=True);return
    from .paths import bootstrap
    bootstrap()
    import cv2,torch
    from graphnav.closedloop import PublicEpisode,PublicCatalog
    from graphnav.dynamic_runtime import NativeObservationBuilder
    from graphnav.stage1_runtime import GraphNavStage1Policy
    from sbfnav.dataset import CityNavDataset,CityReferCatalog
    from sbfnav.planning_state import PlanningStateBuilder
    from sbfnav.mapdata import GROUND_LEVEL
    from .relation_v3_policy import RelationV3Policy
    torch.set_num_threads(2);torch.manual_seed(0);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ds=CityNavDataset(DATA,'train_seen');catalog=CityReferCatalog(DATA/'cityrefer')
    selected=prepare_queries(directory,ds,catalog)
    signature=dict(code_sha256=hashes,roles_sha256=roles,initial_checkpoint_sha256=sha(initial),
        cpu_gates_sha256=sha(root/'cpu_gates.json'),public_episodes_sha256=sha(directory/'public_episodes.json'),
        selected_episodes_sha256=sha(directory/'episodes.json'))
    immutable_json(signature,directory/'signature.json');digest=sha(directory/'signature.json')
    output=directory/f'{arm}.json'
    if output.exists():
        data=json.loads(output.read_text())
        if data['signature_sha256']!=digest or data['episodes_completed']!=32:raise ValueError('Existing arm differs')
        print('Existing completed '+arm+' retained',flush=True);return
    builder=NativeObservationBuilder(PlanningStateBuilder(DATA,PublicCatalog(catalog),canvas_size=384,field_size=28),RGBD,directory/'render_unused')
    require_idle_gpu()
    if arm=='a0':policy=GraphNavStage1Policy(CHECKPOINT,directory/'query_a0')
    else:
        calibration=None
        if arm=='on':
            calibration=directory/'TRAIN_only_uncalibrated_on.json'
            immutable_json(dict(enabled=True,minimum_gain=0.,maximum_harm=1.,temperature=1.,checkpoint_sha256=sha(initial),
                scope='TRAIN_functional_only',gate_final=False),calibration)
        policy=RelationV3Policy(directory/f'query_{arm}',initial,calibration,functional_only=True)
    results=[];captured={}
    def capture_belief(module,args,out):
        values=out.belief.belief_logits.detach().cpu().contiguous()
        if not torch.isfinite(values).all():raise ValueError('Nonfinite functional belief field')
        captured.update(belief_logits_sha256=hashlib.sha256(values.view(torch.uint8).numpy().tobytes()).hexdigest(),
            belief_logits_shape=list(values.shape),belief_logits_dtype=str(values.dtype))
    policy.model.register_forward_hook(capture_belief)
    for number,e in enumerate(selected,1):
        record=ds[e['index']]
        public=PublicEpisode(record.map_name,record.instruction,record.start_pose.copy(),tuple(catalog.referenced_landmarks(record)),tuple(catalog.referenced_landmark_names(record)))
        policy.set_episode(public,builder.transform(record.map_name),catalog)
        trajectory=np.asarray(record.trajectory);legal=np.flatnonzero(np.isfinite(trajectory).all(-1)&(trajectory[:,2]-GROUND_LEVEL[record.map_name]>=10)&(trajectory[:,2]-GROUND_LEVEL[record.map_name]<=100))
        steps=sorted({int(legal[round(q*(len(legal)-1))]) for q in (.25,.75,1.)}) if len(legal) else []
        if not steps:raise ValueError('Selected functional target has no legal sparse states')
        for step in steps:
            observation=builder.build_from_poses(public,trajectory[:step+1]);prediction=policy.predict(observation,record.instruction);d=prediction.diagnostics
            results.append(dict(index=e['index'],step=step,xyz=np.asarray(prediction.xyz).tolist(),should_stop=bool(prediction.should_stop),
                selected_goal_xy=d['selected_goal_xy'],belief_top16_world_xy=d['belief_top16_world_xy'],a2_updated=d.get('a2_updated',False),**captured))
        print(json.dumps(dict(arm=arm,episodes_done=number,views=len(results))),flush=True)
    save(dict(signature_sha256=digest,arm=arm,episodes_completed=32,rows=results),output)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=('a0','off','on','verify'),required=True);args=parser.parse_args()
    if args.arm=='verify':run_arm(args.arm);return
    with (BASE/'gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run_arm(args.arm)


if __name__=='__main__':main()
