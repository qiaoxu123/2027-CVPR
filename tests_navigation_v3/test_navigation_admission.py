import json
from types import SimpleNamespace
import pytest
from experiments.a2_waldo_landmark_belief import relation_v3_navigation_admission as admission
from experiments.a2_waldo_landmark_belief.paths import sha


def fixture(tmp_path,monkeypatch):
    output=tmp_path/'calibration';output.mkdir()
    for name in ('candidate_calibration.json','analysis_signature.json','fit_decision.json'):
        (output/name).write_text('{}')
    bound=dict(candidate_sha256=sha(output/'candidate_calibration.json'),analysis_signature_sha256=sha(output/'analysis_signature.json'),
               fit_decision_sha256=sha(output/'fit_decision.json'))
    (output/'candidate_calibration.sha.json').write_text(json.dumps(bound))
    calls=[];source=SimpleNamespace(state=dict(epoch=25),root=tmp_path)
    monkeypatch.setattr(admission,'FreshConfirmation',lambda root:calls.append('full_training_and_source') or source)
    monkeypatch.setattr(admission,'analysis_signature',lambda s:calls.append('analysis') or {})
    monkeypatch.setattr(admission,'validate_existing',lambda s,a:calls.append('unique_calibration') or {})
    monkeypatch.setattr(admission,'require_candidate',lambda c,d:calls.append('risk'))
    return output,calls,source


def test_uncommitted_or_mutated_candidate_is_rejected_before_source_loading(tmp_path,monkeypatch):
    output,calls,_=fixture(tmp_path,monkeypatch)
    (output/'candidate_calibration.sha.json').unlink()
    with pytest.raises((ValueError,FileNotFoundError)):admission.load_candidate(tmp_path)
    assert calls==[]


def test_root_bound_admission_verifies_sources_and_does_not_create_missing_commit(tmp_path,monkeypatch):
    output,calls,source=fixture(tmp_path,monkeypatch)
    result,state=admission.load_candidate(tmp_path)
    assert result=={} and state==source.state
    assert calls==['full_training_and_source','analysis','unique_calibration','risk']
    calls.clear();(output/'candidate_calibration.json').write_text('{"enabled":true}')
    with pytest.raises(ValueError):admission.load_candidate(tmp_path)
    assert calls==[]


def test_source_validation_failure_prevents_admission(tmp_path,monkeypatch):
    _,calls,_=fixture(tmp_path,monkeypatch)
    def reject(root):raise ValueError('Full25 proof absent')
    monkeypatch.setattr(admission,'FreshConfirmation',reject)
    with pytest.raises(ValueError,match='Full25'):admission.load_candidate(tmp_path)
    assert calls==[]


def test_validator_cannot_silently_replace_the_committed_candidate(tmp_path,monkeypatch):
    _,calls,_=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(admission,'validate_existing',lambda s,a:dict(enabled=True))
    with pytest.raises(ValueError):admission.load_candidate(tmp_path)
    assert 'risk' not in calls
