import pytest
import json
import hashlib
import copy

from experiments.a2_waldo_landmark_belief.reference_v4_preflight import validate_arm_population,validate_runtime_proof


def test_runtime_rows_must_cover_every_frozen_state_exactly_once():
    states = [dict(index=1,steps=[5,9]),dict(index=2,steps=[6])]
    rows = [dict(index=1,step=5),dict(index=1,step=9),dict(index=2,step=6)]
    validate_arm_population(rows,states)
    for changed in (rows[:-1],rows+[rows[0]],[dict(index=8,step=5),*rows[1:]]):
        with pytest.raises(ValueError): validate_arm_population(changed,states)


def runtime_files(tmp_path):
    def write(name,data):
        path=tmp_path/name;path.write_text(json.dumps(data));return hashlib.sha256(path.read_bytes()).hexdigest()
    binding=dict(code_sha256={'model':'a'},roles_sha256='r',feature_manifest_sha256='m',
        feature_signature_sha256='s',feature_build_status_sha256='f',initial_checkpoint_sha256='i',cpu_gates_sha256='c')
    states=[dict(index=i,steps=[0]) for i in range(32)]
    selected=[dict(index=i,map_name='map',object_id=i,grouped_reference_source=i<16) for i in range(32)]
    public=[dict(map_name='map',instruction=f'Go {i}',start_pose=[0,0,10]) for i in range(32)]
    signature=dict(binding,public_episodes_sha256=write('public_episodes.json',public),
        selected_episodes_sha256=write('episodes.json',selected),replay_states_sha256=write('replay_states.json',states))
    digest=write('signature.json',signature);arm_hashes={}
    rows=[dict(index=i,step=0,xyz=[0,0,10],should_stop=False,selected_goal_xy=[0,0],belief_top16_world_xy=[[0,0]],
               belief_logits_sha256='a',belief_logits_shape=[1,4],belief_logits_dtype='torch.float32',a2_updated=False,grouped_references=1) for i in range(32)]
    for name in ('a0','off','on'):
        own=copy.deepcopy(rows)
        if name=='on':
            for row in own:row.update(xyz=[1,0,10],selected_goal_xy=[1,0],a2_updated=True)
        arm_hashes[name]=write(name+'.json',dict(signature_sha256=digest,arm=name,episodes_completed=32,rows=own))
    proof=dict(signature,signature_sha256=digest,status='v4_real_TRAIN_runtime_preflight_passed',arm_sha256=arm_hashes)
    return proof,binding


def test_runtime_proof_checks_recorded_arm_bytes_and_exact_replay_population(tmp_path):
    proof,binding=runtime_files(tmp_path);validate_runtime_proof(tmp_path,proof,binding)
    path=tmp_path/'on.json';data=json.loads(path.read_text());data['rows'].pop();path.write_text(json.dumps(data))
    with pytest.raises(ValueError): validate_runtime_proof(tmp_path,proof,binding)
    # Even if an arm hash is refreshed, missing states must not become valid.
    proof['arm_sha256']['on']=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError): validate_runtime_proof(tmp_path,proof,binding)


def test_runtime_proof_rejects_deleted_arm_or_changed_public_episode_evidence(tmp_path):
    proof,binding=runtime_files(tmp_path);(tmp_path/'off.json').unlink()
    with pytest.raises((ValueError,FileNotFoundError)): validate_runtime_proof(tmp_path,proof,binding)
    proof,binding=runtime_files(tmp_path);(tmp_path/'public_episodes.json').write_text('[]')
    with pytest.raises(ValueError): validate_runtime_proof(tmp_path,proof,binding)
