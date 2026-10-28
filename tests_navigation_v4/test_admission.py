"""Only the unchanged committed, source-verified V4 decision can start a pilot."""
import importlib
import json
from types import SimpleNamespace
import pytest
from experiments.a2_waldo_landmark_belief.paths import sha,save


def module():
    return importlib.import_module('experiments.a2_waldo_landmark_belief.reference_v4_navigation_admission')


def fixture(tmp_path,monkeypatch):
    m=module();output=tmp_path/'calibration';output.mkdir()
    candidate=dict(checkpoint_sha256='checkpoint',feature_manifest_sha256='1'*64,
                   feature_signature_sha256='2'*64,feature_build_status_sha256='3'*64)
    for name,value in [('candidate_calibration.json',candidate),('analysis_signature.json',{}),('fit_decision.json',{})]:save(value,output/name)
    bound=dict(candidate_sha256=sha(output/'candidate_calibration.json'),analysis_signature_sha256=sha(output/'analysis_signature.json'),
               fit_decision_sha256=sha(output/'fit_decision.json'))
    save(bound,output/'candidate_calibration.sha.json')
    calls=[];source=SimpleNamespace(state=dict(candidate,epoch=25),checkpoint_sha256='checkpoint',root=tmp_path)
    monkeypatch.setattr(m,'FreshConfirmation',lambda root:calls.append('full_training_and_source') or source)
    monkeypatch.setattr(m,'analysis_signature',lambda s:calls.append('analysis') or {})
    monkeypatch.setattr(m,'validate_existing',lambda s,a:calls.append('unique_calibration') or candidate)
    monkeypatch.setattr(m,'require_candidate',lambda c,d:calls.append('risk'))
    return m,output,calls,source,candidate


def test_missing_or_changed_commit_is_rejected_before_source_loading(tmp_path,monkeypatch):
    m,output,calls,_,_=fixture(tmp_path,monkeypatch)
    saved=(output/'candidate_calibration.sha.json').read_bytes();(output/'candidate_calibration.sha.json').unlink()
    with pytest.raises((ValueError,FileNotFoundError)):m.load_candidate(tmp_path)
    assert calls==[]
    (output/'candidate_calibration.sha.json').write_bytes(saved);save(dict(enabled=True),output/'candidate_calibration.json')
    with pytest.raises(ValueError):m.load_candidate(tmp_path)
    assert calls==[]


def test_verified_same_candidate_and_new_feature_binding_are_required(tmp_path,monkeypatch):
    m,output,calls,source,candidate=fixture(tmp_path,monkeypatch)
    result,state=m.load_candidate(tmp_path)
    assert result==candidate and state==source.state
    assert calls==['full_training_and_source','analysis','unique_calibration','risk']
    calls.clear();source.state['feature_manifest_sha256']='9'*64
    with pytest.raises(ValueError):m.load_candidate(tmp_path)
    assert 'risk' not in calls


def test_source_failure_or_validator_replacement_prevents_admission(tmp_path,monkeypatch):
    m,_,calls,_,_=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(m,'validate_existing',lambda source,analysis:dict(enabled=True))
    with pytest.raises(ValueError):m.load_candidate(tmp_path)
    assert 'risk' not in calls
    calls.clear()
    def reject(root):raise ValueError('Full25 V4 proof absent')
    monkeypatch.setattr(m,'FreshConfirmation',reject)
    with pytest.raises(ValueError,match='Full25'):m.load_candidate(tmp_path)
    assert calls==[]
