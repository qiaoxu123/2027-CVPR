import copy,json
import pytest
from experiments.a2_waldo_landmark_belief import relation_v3_navigation_freeze as freeze
from experiments.a2_waldo_landmark_belief.paths import sha


def test_final_gate_preserves_every_unique_candidate_parameter(tmp_path):
    candidate=dict(enabled=True,gate_final=False,status='candidate',minimum_gain=.9,maximum_harm=.1,temperature=1.,fixed_action=.25)
    (tmp_path/'pilot').mkdir();(tmp_path/'pilot/report.json').write_text('{}')
    result=freeze.final_parameters(candidate,tmp_path)
    assert result['gate_final'] and result['pilot_report_sha256']==sha(tmp_path/'pilot/report.json')
    assert all(result[k]==candidate[k] for k in candidate if k not in ('gate_final','status'))


def test_frozen_file_manifest_rejects_changed_artifact(tmp_path):
    path=tmp_path/'weights.pt';path.write_bytes(b'original')
    bound={str(path):sha(path)};freeze.check_bound_files(bound)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError):freeze.check_bound_files(bound)


def test_real_pilot_failure_or_incomplete_population_cannot_freeze():
    good=dict(decision=dict(passed=True,episodes=64),benchmark_used=False,
        checkpoint_sha256='model',calibration_candidate_sha256='candidate',code_sha256={'x':'hash'})
    freeze.require_pilot(good,'model','candidate',{'x':'hash'})
    for field,value in [('passed',False),('episodes',63)]:
        bad=copy.deepcopy(good);bad['decision'][field]=value
        with pytest.raises(ValueError):freeze.require_pilot(bad,'model','candidate',{'x':'hash'})
    with pytest.raises(ValueError):freeze.require_pilot(good,'different','candidate',{'x':'hash'})
