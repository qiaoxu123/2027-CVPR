import json
import hashlib
import pytest

from experiments.a2_waldo_landmark_belief import reference_v4_preparation as module


def fixture(tmp_path,monkeypatch):
    directory=tmp_path/'pretraining';directory.mkdir()
    initial=directory/'initial.pt';initial.write_bytes(b'actual initial snapshot')
    log=directory/'cpu_tests.log';log.write_text('558 passed\n')
    assets={'baseline':'a'}
    path=directory/'frozen_assets.json';path.write_text(json.dumps(assets))
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    feature=dict(roles_sha256='r',feature_manifest_sha256='m',feature_signature_sha256='s',feature_build_status_sha256='f')
    binding=dict(feature,code_sha256={'model':'a'},initial_checkpoint_sha256=digest(initial))
    cpu=dict(binding,test_source_sha256={'test':'b'},test_log_sha256=digest(log),frozen_assets_sha256=digest(path))
    monkeypatch.setattr(module,'DIRECTORY',tmp_path)
    monkeypatch.setattr(module,'validate_complete_feature_cache',lambda: feature)
    monkeypatch.setattr(module,'code_hashes',lambda: {'model':'a'})
    monkeypatch.setattr(module,'test_hashes',lambda: {'test':'b'})
    monkeypatch.setattr(module,'runtime_assets',lambda: assets)
    return cpu,binding,log,path


def test_cpu_proof_rechecks_actual_test_log_bytes(tmp_path,monkeypatch):
    cpu,binding,log,_=fixture(tmp_path,monkeypatch)
    assert module.current_binding(cpu)==binding
    log.write_text('1 failed\n')
    with pytest.raises(ValueError): module.current_binding(cpu)


def test_cpu_proof_rechecks_frozen_asset_description(tmp_path,monkeypatch):
    cpu,binding,_,path=fixture(tmp_path,monkeypatch)
    assert module.current_binding(cpu)==binding
    path.write_text(json.dumps({'baseline':'changed'}))
    with pytest.raises(ValueError): module.current_binding(cpu)
