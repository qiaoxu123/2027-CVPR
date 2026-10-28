"""V4 collector lineage cannot admit diagnostic or different-feature states."""
import copy
import importlib
import pytest
from experiments.a2_waldo_landmark_belief.paths import save
from tests_confirmation_v4.test_protocol import live_fixture,binding


def collector():
    return importlib.import_module('experiments.a2_waldo_landmark_belief.reference_v4_confirmation')


def setup(tmp_path,monkeypatch):
    protocol,old,root,_=live_fixture(tmp_path,monkeypatch);protocol.prepare()
    import json
    population=json.loads((root/'confirmation_population.json').read_text())
    state={key:population[key] for key in binding()}
    state.update(config=dict(version='a2_relation_v4_public_roads'),epoch=23,checkpoint_eligible=True)
    checkpoint=root/'training/best.pt';checkpoint.parent.mkdir();checkpoint.write_bytes(b'pinned model bytes')
    save(dict(status='verified training fixture'),root/'training/artifact_verification.json')
    save(dict(sensor='native384'),root/'confirmation_sensor_signature.json')
    return root,state,checkpoint


def test_collection_signature_binds_actual_model_features_and_v4_population(tmp_path,monkeypatch):
    root,state,checkpoint=setup(tmp_path,monkeypatch)
    value=collector().collection_signature(root,state,checkpoint)
    assert value['feature_manifest_sha256']==state['feature_manifest_sha256']
    assert value['feature_signature_sha256']==state['feature_signature_sha256']
    assert value['feature_build_status_sha256']==state['feature_build_status_sha256']
    assert 'V4 shadowOFF' in value['observation_source']
    checkpoint.write_bytes(b'other model bytes')
    assert collector().collection_signature(root,state,checkpoint)['checkpoint_sha256']!=value['checkpoint_sha256']


@pytest.mark.parametrize('change',[
    dict(checkpoint_eligible=False),dict(epoch=0),dict(epoch=20),
    dict(scope='TRAIN_functional_only'),dict(config=dict(version='a2_relation_v3')),
    dict(feature_manifest_sha256='9'*64),dict(feature_signature_sha256='9'*64),
    dict(feature_build_status_sha256='9'*64),dict(roles_sha256='9'*64),
    dict(code_sha256={'old.py':'9'*64}),dict(initial_checkpoint_sha256='9'*64),
])
def test_collection_rejects_diagnostic_old_version_or_rebound_features(tmp_path,monkeypatch,change):
    root,state,checkpoint=setup(tmp_path,monkeypatch);state.update(copy.deepcopy(change))
    with pytest.raises(ValueError):collector().collection_signature(root,state,checkpoint)


def test_no_eligible_model_leaves_independent_population_unconsumed(monkeypatch):
    m=collector()
    monkeypatch.setattr(m,'require_idle_gpu',lambda **kwargs:None)
    monkeypatch.setattr(m,'checked_state',lambda checkpoint:dict(checkpoint_eligible=False))
    def forbidden():raise AssertionError('No data preparation or collection before eligible full training')
    monkeypatch.setattr(m,'prepare',forbidden)
    with pytest.raises(ValueError,match='do not consume'):m.collect()
