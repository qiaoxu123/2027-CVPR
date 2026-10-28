import json
import pytest
from experiments.a2_waldo_landmark_belief.relation_v3_pipeline import phase_done,is_busy_rejection


def test_pipeline_retains_only_matching_completed_proofs(tmp_path):
    hashes={'code':'sha'};role='role'
    assert not phase_done('cuda',tmp_path,hashes,role)
    path=tmp_path/'cuda_smoke.json'
    data=dict(status='v3_real_TRAIN_CUDA_backward_passed',code_sha256=hashes,roles_sha256=role)
    path.write_text(json.dumps(data));assert phase_done('cuda',tmp_path,hashes,role)
    data['code_sha256']={'other':'sha'};path.write_text(json.dumps(data))
    with pytest.raises(ValueError):phase_done('cuda',tmp_path,hashes,role)


def test_retry_uses_only_current_attempt_last_error(tmp_path):
    path=tmp_path/'job.log';old='ValueError: GPU occupied; no parallel A2 GPU work allowed\n'
    path.write_text(old);offset=path.stat().st_size
    with path.open('a') as stream:stream.write('ValueError: corrupted checkpoint\n')
    assert not is_busy_rejection(path,offset)
    offset=path.stat().st_size
    with path.open('a') as stream:stream.write(old)
    assert is_busy_rejection(path,offset)
