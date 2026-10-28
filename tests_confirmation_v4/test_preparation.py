"""Post-training CPU admission must bind actual logs and current source inputs."""
import importlib
import json
import pytest
from experiments.a2_waldo_landmark_belief.paths import save,sha
from tests_confirmation_v4.test_collection import setup


def module():
    return importlib.import_module('experiments.a2_waldo_landmark_belief.reference_v4_confirmation_preparation')


def fixture(tmp_path,monkeypatch):
    root,state,_=setup(tmp_path,monkeypatch);m=module()
    save(dict(status='original_cpu_gate_fixture'),root/'cpu_gates.json')
    monkeypatch.setattr(m,'BASE',root.parent)
    signature=m.cpu_signature(root)
    log=root/'posttraining_cpu/cpu_tests.log';log.parent.mkdir();log.write_text('621 passed in 1s\n')
    proof=dict(signature,status='v4_posttraining_CPU_gates_passed',test_log_sha256=sha(log),
               unit_test_summary='621 passed in 1s',no_CUDA_used=True,no_new_confirmation_data_consumed=True)
    save(proof,root/'posttraining_cpu_gates.json')
    return m,root,signature


def test_current_gate_rejects_changed_log_or_training_proof(tmp_path,monkeypatch):
    m,root,_=fixture(tmp_path,monkeypatch);m.require_cpu_gates(root)
    log=root/'posttraining_cpu/cpu_tests.log';old=log.read_bytes();log.write_text('invented pass')
    with pytest.raises(ValueError):m.require_cpu_gates(root)
    log.write_bytes(old);save(dict(status='replaced training proof'),root/'cpu_gates.json')
    with pytest.raises(ValueError):m.require_cpu_gates(root)


@pytest.mark.parametrize('name',['confirmation_population.json','confirmation_protocol.json'])
def test_current_gate_rejects_changed_population_or_protocol(tmp_path,monkeypatch,name):
    m,root,_=fixture(tmp_path,monkeypatch);path=root/name
    value=json.loads(path.read_text());value['unexpected_change']=True;save(value,path)
    with pytest.raises(ValueError):m.require_cpu_gates(root)


def test_gpu_or_data_consumption_claim_cannot_be_accepted_as_cpu_gate(tmp_path,monkeypatch):
    m,root,_=fixture(tmp_path,monkeypatch);path=root/'posttraining_cpu_gates.json';original=json.loads(path.read_text())
    for key in ('no_CUDA_used','no_new_confirmation_data_consumed'):
        proof=dict(original);proof[key]=False;save(proof,path)
        with pytest.raises(ValueError):m.require_cpu_gates(root)


@pytest.mark.parametrize('marker',['confirmation/signature.json','calibration/fit_decision.json'])
def test_stale_gate_cannot_be_regenerated_after_observation_or_fit_choice(tmp_path,monkeypatch,marker):
    m,root,_=fixture(tmp_path,monkeypatch);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    save({},root/marker)
    (root/'posttraining_cpu/cpu_tests.log').write_text('stale proof log')
    original=(root/'posttraining_cpu_gates.json').read_bytes()
    with pytest.raises(ValueError,match='Cannot replace'):m.cpu_gates()
    assert (root/'posttraining_cpu_gates.json').read_bytes()==original
    assert (root/'posttraining_cpu/cpu_tests.log').read_text()=='stale proof log'
