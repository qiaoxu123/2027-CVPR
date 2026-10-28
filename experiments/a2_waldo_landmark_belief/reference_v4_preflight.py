"""Serial native384 TRAIN A0/OFF/ON replay with V4 public reference features."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np

from .paths import DATA,RGBD,CHECKPOINT,sha,save
from .reference_overlay_context import BASE,DEFAULT_OUTPUT
from .reference_overlay import overlay_filename,read_reference_overlay
from .reference_v4_preparation import current_binding,DIRECTORY
from .relation_v3_train import require_idle_gpu
from .relation_v3_preflight import check_records
from .revision_confirmation import immutable_json


def validate_arm_population(rows,states):
    expected={(int(record['index']),int(step)) for record in states for step in record['steps']}
    actual=[(int(record['index']),int(record['step'])) for record in rows]
    if not expected or len(actual)!=len(set(actual)) or set(actual)!=expected:
        raise ValueError('Functional arm does not cover the frozen replay states exactly once')


def validate_runtime_proof(directory,proof,binding):
    """Recheck actual replay evidence at admission, not only summary booleans."""
    directory=Path(directory)
    signature_path=directory/'signature.json'
    signature=json.loads(signature_path.read_text())
    hashes={'public_episodes_sha256':'public_episodes.json','selected_episodes_sha256':'episodes.json',
            'replay_states_sha256':'replay_states.json'}
    if (proof.get('status')!='v4_real_TRAIN_runtime_preflight_passed'
        or proof.get('signature_sha256')!=sha(signature_path)
        or set(signature)!=set(binding)|set(hashes)
        or any(signature.get(key)!=value or proof.get(key)!=value for key,value in binding.items())
        or any(proof.get(key)!=signature.get(key) or sha(directory/name)!=signature.get(key) for key,name in hashes.items())):
        raise ValueError('V4 runtime proof source/signature/replay evidence changed')
    selected=json.loads((directory/'episodes.json').read_text())
    public=json.loads((directory/'public_episodes.json').read_text())
    states=json.loads((directory/'replay_states.json').read_text())
    if (len(selected)!=32 or len(public)!=32 or len(states)!=32
        or len({(entry['map_name'],entry['object_id']) for entry in selected})!=32
        or [entry['index'] for entry in selected]!=[entry['index'] for entry in states]
        or sum(bool(entry['grouped_reference_source']) for entry in selected)!=16):
        raise ValueError('V4 runtime proof TRAIN target cohort changed')
    arm_hashes=proof.get('arm_sha256',{})
    if set(arm_hashes)!={'a0','off','on'}:
        raise ValueError('All three V4 runtime arm hashes required')
    arms=[];grouped=0
    for name in ('a0','off','on'):
        path=directory/f'{name}.json'
        if sha(path)!=arm_hashes[name]:
            raise ValueError('V4 runtime arm bytes changed')
        data=json.loads(path.read_text())
        if data.get('signature_sha256')!=sha(signature_path) or data.get('arm')!=name or data.get('episodes_completed')!=32:
            raise ValueError('V4 runtime arm provenance changed')
        validate_arm_population(data['rows'],states);arms.append(data['rows'])
        if name!='a0':grouped+=sum(row['grouped_references'] for row in data['rows'])
    if not grouped: raise ValueError('V4 grouped-reference runtime was not exercised')
    return check_records(*arms)


def prepare_queries(directory,ds,catalog,ground_level):
    manifest=json.loads((DEFAULT_OUTPUT/'manifest.json').read_text())
    roles=json.loads((BASE/'revision_v3/roles.json').read_text())
    allowed={(entry['map_name'],entry['object_id']) for entry in roles['target_roles']['train']}
    entries=[entry for entry in manifest['entries'] if entry['role']=='train']
    entries.sort(key=lambda entry:hashlib.sha256(('reference-v4-functional:0:'+overlay_filename(entry)).encode()).digest())
    selected=[];seen=set();counts={True:0,False:0}
    for entry in entries:
        target=(entry['map_name'],entry['object_id'])
        if target in seen: continue
        overlay=read_reference_overlay(DEFAULT_OUTPUT/'views',entry,role='train',allowed_targets=allowed)
        if not overlay['network']['pair_valid'].any(): continue
        grouped=any('region' in ref for ref in overlay['meta']['references'])
        if counts[grouped]>=16: continue
        selected.append(dict(index=entry['episode'],map_name=entry['map_name'],object_id=entry['object_id'],
                             grouped_reference_source=grouped,parent_sha256=entry['sha256'],overlay_sha256=entry['overlay_sha256']))
        seen.add(target);counts[grouped]+=1
        if len(selected)==32: break
    if len(selected)!=32 or counts!={True:16,False:16}:
        raise ValueError('All32 distinct TRAIN targets including16 grouped references required')
    public=[];states=[]
    for entry in selected:
        record=ds[entry['index']]
        if (record.map_name,int(record.object_id))!=(entry['map_name'],entry['object_id']):
            raise ValueError('Functional TRAIN source identity changed')
        public.append(dict(map_name=record.map_name,instruction=record.instruction,start_pose=record.start_pose.tolist()))
        trajectory=np.asarray(record.trajectory)
        legal=np.flatnonzero(np.isfinite(trajectory).all(-1)&(trajectory[:,2]-ground_level[record.map_name]>=10)
                             &(trajectory[:,2]-ground_level[record.map_name]<=100))
        steps=sorted({int(legal[round(q*(len(legal)-1))]) for q in (.25,.75,1.)}) if len(legal) else []
        if not steps: raise ValueError('Functional TRAIN target has no legal sparse poses')
        states.append(dict(index=entry['index'],steps=steps))
    immutable_json(selected,directory/'episodes.json');immutable_json(public,directory/'public_episodes.json')
    immutable_json(states,directory/'replay_states.json')
    return selected,states


def run_arm(arm):
    directory=DIRECTORY/'preflight';directory.mkdir(parents=True,exist_ok=True)
    binding=current_binding();initial=DIRECTORY/'pretraining/initial.pt'
    if arm=='verify':
        signature=json.loads((directory/'signature.json').read_text())
        if (any(signature.get(key)!=value for key,value in binding.items())
            or signature['cpu_gates_sha256']!=sha(DIRECTORY/'cpu_gates.json')
            or signature['public_episodes_sha256']!=sha(directory/'public_episodes.json')
            or signature['selected_episodes_sha256']!=sha(directory/'episodes.json')
            or signature['replay_states_sha256']!=sha(directory/'replay_states.json')):
            raise ValueError('V4 functional signature or replay states changed')
        states=json.loads((directory/'replay_states.json').read_text())
        selected=json.loads((directory/'episodes.json').read_text())
        if len(selected)!=32 or len({(e['map_name'],e['object_id']) for e in selected})!=32:
            raise ValueError('All32 functional TRAIN target identities required')
        rows=[];arms={};grouped=0;arm_hashes={}
        for name in ('a0','off','on'):
            path=directory/f'{name}.json';data=json.loads(path.read_text())
            if data['signature_sha256']!=sha(directory/'signature.json') or data['episodes_completed']!=32 or data['arm']!=name:
                raise ValueError('Incomplete or foreign V4 functional arm')
            validate_arm_population(data['rows'],states)
            rows.append(data['rows']);arms[name.upper()]=data['episodes_completed'];arm_hashes[name]=sha(path)
            if name!='a0': grouped+=sum(record['grouped_references'] for record in data['rows'])
        result=check_records(*rows)
        if not grouped: raise ValueError('Actual grouped-reference runtime path was not exercised')
        report=dict(signature,status='v4_real_TRAIN_runtime_preflight_passed',arm_counts=arms,arm_sha256=arm_hashes,
            signature_sha256=sha(directory/'signature.json'),
            exact_a0_off_parity=result['off_parity_passed'],original_controller_verified=result['controller_coupling_passed'],
            functional_train_only=True,benchmark_used=False,grouped_reference_views=grouped,
            diagnostics=result,scope='Sparse TRAIN causal replay, not navigation gain')
        validate_runtime_proof(directory,report,dict(binding,cpu_gates_sha256=sha(DIRECTORY/'cpu_gates.json')))
        save(report,DIRECTORY/'runtime_preflight.json');print(json.dumps(report),flush=True);return
    from .paths import bootstrap
    bootstrap()
    import cv2,torch
    from graphnav.closedloop import PublicEpisode,PublicCatalog
    from graphnav.dynamic_runtime import NativeObservationBuilder
    from graphnav.stage1_runtime import GraphNavStage1Policy
    from sbfnav.dataset import CityNavDataset,CityReferCatalog
    from sbfnav.planning_state import PlanningStateBuilder
    from sbfnav.mapdata import GROUND_LEVEL
    from .reference_v4_policy import ReferenceV4Policy
    torch.set_num_threads(2);torch.manual_seed(0);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ds=CityNavDataset(DATA,'train_seen');catalog=CityReferCatalog(DATA/'cityrefer')
    selected,states=prepare_queries(directory,ds,catalog,GROUND_LEVEL)
    signature=dict(binding,cpu_gates_sha256=sha(DIRECTORY/'cpu_gates.json'),
        public_episodes_sha256=sha(directory/'public_episodes.json'),selected_episodes_sha256=sha(directory/'episodes.json'),
        replay_states_sha256=sha(directory/'replay_states.json'))
    immutable_json(signature,directory/'signature.json');digest=sha(directory/'signature.json')
    output=directory/f'{arm}.json'
    if output.exists():
        data=json.loads(output.read_text())
        if data['signature_sha256']!=digest or data['episodes_completed']!=32 or data['arm']!=arm:
            raise ValueError('Existing V4 functional arm differs')
        validate_arm_population(data['rows'],states)
        print('Existing complete V4 '+arm+' retained',flush=True);return
    builder=NativeObservationBuilder(PlanningStateBuilder(DATA,PublicCatalog(catalog),canvas_size=384,field_size=28),
                                     RGBD,directory/'render_unused')
    require_idle_gpu()
    if arm=='a0': policy=GraphNavStage1Policy(CHECKPOINT,directory/'query_a0')
    else:
        calibration=None
        if arm=='on':
            calibration=directory/'TRAIN_only_uncalibrated_on.json'
            immutable_json(dict(enabled=True,minimum_gain=0.,maximum_harm=1.,temperature=1.,checkpoint_sha256=sha(initial),
                scope='TRAIN_functional_only',gate_final=False),calibration)
        policy=ReferenceV4Policy(directory/f'query_{arm}',initial,calibration,functional_only=True)
    captured={};results=[]
    def capture_belief(module,args,out):
        values=out.belief.belief_logits.detach().cpu().contiguous()
        if not torch.isfinite(values).all(): raise ValueError('Nonfinite V4 functional belief')
        captured.update(belief_logits_sha256=hashlib.sha256(values.view(torch.uint8).numpy().tobytes()).hexdigest(),
                        belief_logits_shape=list(values.shape),belief_logits_dtype=str(values.dtype))
    policy.model.register_forward_hook(capture_belief)
    for number,(entry,replay) in enumerate(zip(selected,states),1):
        record=ds[entry['index']]
        public=PublicEpisode(record.map_name,record.instruction,record.start_pose.copy(),
                             tuple(catalog.referenced_landmarks(record)),tuple(catalog.referenced_landmark_names(record)))
        policy.set_episode(public,builder.transform(record.map_name),catalog)
        trajectory=np.asarray(record.trajectory)
        for step in replay['steps']:
            observation=builder.build_from_poses(public,trajectory[:step+1]);prediction=policy.predict(observation,record.instruction)
            diagnostics=prediction.diagnostics;meta=getattr(policy,'meta',None)
            grouped=sum('region' in ref for ref in meta['references']) if meta else 0
            results.append(dict(index=entry['index'],step=step,xyz=np.asarray(prediction.xyz).tolist(),should_stop=bool(prediction.should_stop),
                selected_goal_xy=diagnostics['selected_goal_xy'],belief_top16_world_xy=diagnostics['belief_top16_world_xy'],
                a2_updated=diagnostics.get('a2_updated',False),grouped_references=grouped,**captured))
        print(json.dumps(dict(arm=arm,episodes_done=number,views=len(results))),flush=True)
    validate_arm_population(results,states)
    if current_binding()!=binding: raise ValueError('V4 sources/assets changed during runtime check')
    save(dict(signature_sha256=digest,arm=arm,episodes_completed=32,rows=results),output)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=('a0','off','on','verify'),required=True);args=parser.parse_args()
    if args.arm=='verify': run_arm(args.arm);return
    with (BASE/'gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run_arm(args.arm)


if __name__=='__main__': main()
