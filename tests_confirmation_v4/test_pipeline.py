"""Serial handoff requires authoritative terminal training, never partial files."""
import importlib
import json
from types import SimpleNamespace
import pytest
from experiments.a2_waldo_landmark_belief.paths import save


def module():
    return importlib.import_module('experiments.a2_waldo_landmark_belief.reference_v4_confirmation_pipeline')


def service(**updates):
    result=dict(LoadState='loaded',ActiveState='active',SubState='running',MainPID='17',ExecMainStatus='0',Result='success')
    result.update(updates);return result


def test_active_training_is_waited_for_even_if_checkpoint_files_exist():
    m=module();assert not m.predecessor_ready(service(),True)
    assert m.predecessor_ready(service(MainPID='0',SubState='exited'),True)
    with pytest.raises(ValueError):m.predecessor_ready(service(MainPID='0',SubState='exited'),False)
    with pytest.raises(ValueError):m.predecessor_ready(service(LoadState='not-found'),True)
    with pytest.raises(ValueError):m.predecessor_ready(service(ActiveState='failed',ExecMainStatus='1',Result='exit-code'),True)


def test_service_read_failure_is_observation_failure_not_terminal_success(monkeypatch):
    m=module()
    monkeypatch.setattr(m.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=1,stdout='',stderr='temporary dbus error'))
    with pytest.raises(m.ObservationUnavailable):m.read_predecessor()
    monkeypatch.setattr(m.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=1,stdout='LoadState=not-found\n',stderr='missing'))
    with pytest.raises(ValueError):m.predecessor_ready(m.read_predecessor(),True)


def test_pipeline_waits_then_refuses_ineligible_model_without_gpu_work(tmp_path,monkeypatch):
    m=module();root=tmp_path/'revision_v4';root.mkdir()
    save({},root/'training/artifact_verification.json');save({},root/'training/curve_manifest.json')
    monkeypatch.setattr(m,'require_cpu_gates',lambda root:None)
    monkeypatch.setattr(m,'cpu_signature',lambda root:{'unchanged':'source'})
    observations=iter([service(),service(MainPID='0',SubState='exited')])
    monkeypatch.setattr(m,'read_predecessor',lambda:next(observations))
    sleeps=[];monkeypatch.setattr(m.time,'sleep',lambda seconds:sleeps.append(seconds))
    monkeypatch.setattr(m,'admitted_training',lambda root:dict(checkpoint_eligible=False))
    def forbidden(*a,**k):raise AssertionError('Independent data or GPU used without eligible complete training')
    monkeypatch.setattr(m.subprocess,'run',forbidden);monkeypatch.setattr(m,'FreshConfirmation',forbidden)
    m.run_pipeline(root=root,poll_seconds=1)
    status=json.loads((root/'post_pipeline_status.json').read_text())
    assert sleeps==[1] and status['state']=='full_training_no_eligible_DEV_gate_confirmation_unconsumed'
    assert status['objective_achieved'] is False and not status['full_benchmark_completed']


def test_transient_observation_retries_same_predecessor_without_launching_work(tmp_path,monkeypatch):
    m=module();root=tmp_path/'revision_v4';root.mkdir()
    save({},root/'training/artifact_verification.json');save({},root/'training/curve_manifest.json')
    monkeypatch.setattr(m,'require_cpu_gates',lambda root:None);monkeypatch.setattr(m,'cpu_signature',lambda root:{})
    calls=[]
    def observe():
        calls.append(True)
        if len(calls)==1:raise m.ObservationUnavailable('transient read timeout')
        return service(MainPID='0',SubState='exited')
    monkeypatch.setattr(m,'read_predecessor',observe);monkeypatch.setattr(m.time,'sleep',lambda seconds:None)
    monkeypatch.setattr(m,'admitted_training',lambda root:dict(checkpoint_eligible=False))
    monkeypatch.setattr(m.subprocess,'run',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Unexpected child launch')))
    m.run_pipeline(root=root,poll_seconds=1)
    assert len(calls)==2


def test_no_fit_choice_never_launches_replay_or_verify(tmp_path):
    save(dict(selected_parameters=None),tmp_path/'calibration/fit_decision.json')
    def forbidden(*a,**k):raise AssertionError('Heldout phase launched without a choice')
    result=module().after_fit(tmp_path,forbidden,lambda:dict(enabled=False,verification=None,verification_performance_evaluated=False))
    assert result=='FIT_no_candidate_VERIFY_unconsumed'


def test_unique_fit_choice_orders_replay_before_single_verify(tmp_path):
    save(dict(selected_parameters=dict(minimum_gain=.8,maximum_harm=.1,temperature=1.)),tmp_path/'calibration/fit_decision.json')
    phases=[]
    result=module().after_fit(tmp_path,phases.append,lambda:dict(enabled=False,verification=dict(groups=209),verification_performance_evaluated=True))
    assert phases==['replay','verify'] and result=='VERIFY_rejected_no_forced_enable'


def test_eligible_pipeline_runs_children_serially_and_waits_for_gpu_idle(tmp_path,monkeypatch):
    m=module();root=tmp_path/'revision_v4';root.mkdir()
    save({},root/'training/artifact_verification.json');save({},root/'training/curve_manifest.json')
    monkeypatch.setattr(m,'require_cpu_gates',lambda root:None);monkeypatch.setattr(m,'cpu_signature',lambda root:{})
    monkeypatch.setattr(m,'read_predecessor',lambda:service(MainPID='0',SubState='exited'))
    monkeypatch.setattr(m,'admitted_training',lambda root:dict(checkpoint_eligible=True))
    occupancy=iter([[999],[],[]]);monkeypatch.setattr(m,'gpu_pids',lambda:next(occupancy))
    sleeps=[];monkeypatch.setattr(m.time,'sleep',lambda seconds:sleeps.append(seconds))
    phases=[];validated=[]
    def run(command,**kwargs):
        name=command[3].rsplit('.',1)[-1]
        phase={'reference_v4_confirmation':'collect','reference_v4_sealed_replay':'replay'}.get(name)
        if phase is None:phase=command[-1]
        phases.append(phase)
        if phase in ('fit','verify'):assert kwargs['env']['CUDA_VISIBLE_DEVICES']==''
        if phase=='fit':save(dict(selected_parameters=dict(minimum_gain=.8,maximum_harm=.1,temperature=1.)),root/'calibration/fit_decision.json')
        if phase=='verify':save({},root/'calibration/candidate_calibration.json')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(m.subprocess,'run',run)
    monkeypatch.setattr(m,'validate_completed_phase',lambda root,name:validated.append(name))
    monkeypatch.setattr(m,'validated_result',lambda root:dict(enabled=False,verification=dict(groups=209),verification_performance_evaluated=True))
    m.run_pipeline(root=root,poll_seconds=1)
    assert phases==['collect','fit','replay','verify'] and validated==phases and sleeps==[1]
    final=json.loads((root/'post_pipeline_status.json').read_text())
    assert final['state']=='VERIFY_rejected_no_forced_enable' and not final['full_benchmark_completed']
