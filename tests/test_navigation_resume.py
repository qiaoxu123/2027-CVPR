import json
import pytest
from graphnav.checkpointing import file_sha256
from graphnav.navigation_resume import prepare_resume,load_prefix


def fixture(tmp_path):
    native=tmp_path/'native.pt';native.write_bytes(b'native');base=tmp_path/'base.pt';base.write_bytes(b'base')
    frozen=dict(native_checkpoint_sha=file_sha256(native),base_checkpoint_sha=file_sha256(base),native_history_window=8,native_control='real',head_execution='guarded-blend',stop_threshold=.3)
    freeze=tmp_path/'freeze.json';freeze.write_text(json.dumps(frozen))
    manifest=[dict(index=i,episode_id=['map',i],start_pose=[0.,0.,30.,0.]) for i in range(3)]
    (tmp_path/'episodes.json').write_text(json.dumps(manifest))
    d=dict(observation_step=0,native_object_checkpoint_sha=frozen['native_checkpoint_sha'],native_object_history_window=8,native_object_control='real',execution='guarded-blend',active_observation=False)
    row=dict(index=0,episode_id=['map',0],poses=[[0.,0.,30.,0.]],step_diagnostics=[d],decision_seconds=[.1],termination='horizon')
    (tmp_path/'predictions.jsonl').write_text(json.dumps(row)+'\n')
    return freeze,native,base,manifest


def test_resume_preserves_prefix_and_rejects_selection_changes(tmp_path):
    freeze,native,base,manifest=fixture(tmp_path)
    before=(tmp_path/'predictions.jsonl').read_bytes()
    contract=prepare_resume(tmp_path,freeze,native,base,8,'real')
    rows,loaded=load_prefix(tmp_path,manifest)
    assert len(rows)==1 and loaded==contract
    assert (tmp_path/contract['prefix_backup']).read_bytes()==before
    assert (tmp_path/'predictions.jsonl').read_bytes()==before
    with pytest.raises(ValueError,match='dataset/episode selection'):
        load_prefix(tmp_path,manifest[::-1])


def test_resume_rejects_changed_controls_truncated_episode_or_completed_run(tmp_path):
    freeze,native,base,manifest=fixture(tmp_path)
    with pytest.raises(ValueError,match='frozen candidate'):
        prepare_resume(tmp_path,freeze,native,base,4,'real')
    path=tmp_path/'predictions.jsonl';path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError,match='newline-terminated'):
        prepare_resume(tmp_path,freeze,native,base,8,'real')
    (tmp_path/'metrics.json').write_text('{}')
    with pytest.raises(ValueError,match='completed evaluation'):
        prepare_resume(tmp_path,freeze,native,base,8,'real')
