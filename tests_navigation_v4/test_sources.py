"""Navigation CPU proof binds current implementation, tests, metadata and log."""
import importlib
import json
import pytest
from experiments.a2_waldo_landmark_belief.paths import sha,save
from tests_navigation_v4.test_protocol import fixture as protocol_fixture


def module():
    return importlib.import_module('experiments.a2_waldo_landmark_belief.reference_v4_navigation_sources')


def fixture(tmp_path,monkeypatch):
    protocol,_,root=protocol_fixture(tmp_path,monkeypatch);protocol.prepare_pilot(root)
    m=module();monkeypatch.setattr(m,'BASE',root.parent)
    monkeypatch.setattr(m,'require_post_cpu_gates',lambda root:None)
    save(dict(status='post_cpu_fixture'),root/'posttraining_cpu_gates.json')
    log=root/'navigation_cpu_tests.log';log.write_text('696 passed in 1s\n')
    proof=dict(m.cpu_signature(root),status='v4_real_navigation_CPU_gates_passed',test_log_sha256=sha(log),
        unit_test_summary='696 passed in 1s',no_GPU_used=True,no_navigation_observations=True)
    save(proof,root/'navigation_cpu_gates.json')
    return m,root


def test_navigation_proof_requires_actual_log_and_current_post_proof(tmp_path,monkeypatch):
    m,root=fixture(tmp_path,monkeypatch);m.require_cpu_gates(root)
    log=root/'navigation_cpu_tests.log';old=log.read_bytes();log.write_text('different')
    with pytest.raises(ValueError):m.require_cpu_gates(root)
    log.write_bytes(old);save(dict(status='changed'),root/'posttraining_cpu_gates.json')
    with pytest.raises(ValueError):m.require_cpu_gates(root)


@pytest.mark.parametrize('name',['navigation_pilot_population.json','navigation_protocol.json'])
def test_navigation_proof_rejects_replaced_population_or_protocol(tmp_path,monkeypatch,name):
    m,root=fixture(tmp_path,monkeypatch);path=root/name;value=json.loads(path.read_text())
    value['unexpected_change']=True;save(value,path)
    with pytest.raises(ValueError):m.require_cpu_gates(root)


@pytest.mark.parametrize('marker',['pilot/A0/signature.json','evaluation/val_seen/A0/signature.json','frozen.json'])
def test_stale_navigation_proof_cannot_be_replaced_after_execution_or_freeze(tmp_path,monkeypatch,marker):
    m,root=fixture(tmp_path,monkeypatch);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(m,'prepare_pilot',lambda root:None)
    (root/'navigation_cpu_tests.log').write_text('stale proof');save({},root/marker)
    original=(root/'navigation_cpu_gates.json').read_bytes()
    with pytest.raises(ValueError,match='Cannot replace'):m.cpu_gates(root=root)
    assert (root/'navigation_cpu_gates.json').read_bytes()==original
