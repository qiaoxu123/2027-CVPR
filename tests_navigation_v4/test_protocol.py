"""Navigation population is fixed before model feedback; V4 admission is strict."""
import copy
import importlib
import json
import pytest
from tests_confirmation_v4.test_protocol import live_fixture
from tests_navigation_v3.test_navigation_protocol import candidate as old_candidate


def module():
    return importlib.import_module('experiments.a2_waldo_landmark_belief.reference_v4_navigation_protocol')


def fixture(tmp_path,monkeypatch):
    protocol,old,root,_=live_fixture(tmp_path,monkeypatch);protocol.prepare();m=module()
    from experiments.a2_waldo_landmark_belief import reference_v4_preparation
    monkeypatch.setattr(m,'BASE',root.parent)
    monkeypatch.setattr(m,'current_binding',reference_v4_preparation.current_binding)
    monkeypatch.setattr(m,'require_cpu_gates',lambda path:None)
    monkeypatch.setattr(m,'verify_opaque_seal',lambda seal:None)
    return m,old,root


def test_metadata_keeps_original64_queries_and_does_not_touch_old_files(tmp_path,monkeypatch):
    m,old,root=fixture(tmp_path,monkeypatch)
    originals={p:p.read_bytes() for p in old.iterdir()}
    result=m.prepare_pilot(root)
    population=json.loads((root/'navigation_pilot_population.json').read_text())
    assert len(population['entries'])==64
    assert len({e['index'] for e in population['entries']})==64
    assert all(e['fold']=='verify' for e in population['entries'])
    assert result['tensor_deserializations']==0 and not result['real_pilot_completed']
    assert all(p.read_bytes()==data for p,data in originals.items())
    assert population['signature']['feature_manifest_sha256']=='3'*64
    before=(root/'navigation_pilot_population.json').read_bytes();m.prepare_pilot(root)
    assert (root/'navigation_pilot_population.json').read_bytes()==before


def test_changed_confirmation_or_benchmark_counts_cannot_prepare_navigation(tmp_path,monkeypatch):
    m,_,root=fixture(tmp_path,monkeypatch)
    path=root/'confirmation_population.json';population=json.loads(path.read_text())
    population['fresh_entries'][0]['index']=-1;path.write_text(json.dumps(population))
    with pytest.raises(ValueError):m.prepare_pilot(root)
    assert not (root/'navigation_pilot_population.json').exists()
    m,_,root=fixture(tmp_path/'other',monkeypatch)
    config=copy.deepcopy(m.CONFIG);config['full_counts']['val_seen']=128
    monkeypatch.setattr(m,'CONFIG',config)
    with pytest.raises(ValueError):m.prepare_pilot(root)
    assert not (root/'navigation_pilot_population.json').exists()


def test_pilot_requires_new_verified_candidate_and_refuses_old_version_or_failure():
    m=module();candidate,decision=old_candidate()
    with pytest.raises(ValueError):m.require_candidate(candidate,decision)
    candidate['status']='v4_candidate_requires_real_closed_loop_pilot';m.require_candidate(candidate,decision)
    for field,value in [('enabled',False),('gate_final',True),('verify_groups',208),('fixed_action',.5),('minimum_gain',.7)]:
        changed=copy.deepcopy(candidate);changed[field]=value
        with pytest.raises(ValueError):m.require_candidate(changed,decision)
    changed=copy.deepcopy(candidate);changed['verification']['gain_ci95'][0]=0.
    with pytest.raises(ValueError):m.require_candidate(changed,decision)
