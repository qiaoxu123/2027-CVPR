"""The V4 native/guest handoff cannot promote stale or partial outcomes."""
import copy
import importlib
import pytest


def module():
    return importlib.import_module('experiments.a2_waldo_landmark_belief.reference_v4_navigation_pipeline')


def identity():
    unit='graphnav-a2-reference-v4-confirmation-20261029.service'
    return dict(units={unit:dict(MainPID='31',InvocationID='a'*32)},
                queue=dict(pid=32,signature_sha256='b'*64))


def service(**changes):
    value=dict(LoadState='loaded',ActiveState='active',SubState='running',MainPID='31',ExecMainPID='31',
               ExecMainStatus='0',Result='success',InvocationID='a'*32)
    value.update(changes);return value


def record(state='calibration_passed_real_pilot_pending'):
    return dict(pid=32,phase='finished',state=state,signature_sha256='b'*64)


def test_live_predecessor_wins_over_finished_record_and_native_guest_pids_can_differ():
    m=module();assert m.predecessor_outcome(service(),record(),identity()) is None
    assert m.predecessor_outcome(service(MainPID='0',SubState='exited'),record(),identity())==m.ADMITTED
    for change in [dict(ExecMainPID='99'),dict(InvocationID='c'*32),dict(ActiveState='failed',Result='exit-code',ExecMainStatus='1')]:
        with pytest.raises(ValueError):m.predecessor_outcome(service(MainPID='0',SubState='exited',**change),record(),identity())
    wrong=record();wrong['pid']=31
    with pytest.raises(ValueError):m.predecessor_outcome(service(MainPID='0',SubState='exited'),wrong,identity())
    wrong=record();wrong['signature_sha256']='c'*64
    with pytest.raises(ValueError):m.predecessor_outcome(service(MainPID='0',SubState='exited'),wrong,identity())


def test_rejected_confirmation_launches_nothing_and_pilot_failure_prevents_formal_splits():
    m=module();called=[]
    assert m.execute_sequence('VERIFY_rejected_no_forced_enable',called.append)=='internal_confirmation_not_admitted'
    assert called==[]
    def run(name):
        called.append(name);return dict(decision=dict(passed=False))
    assert m.execute_sequence(m.ADMITTED,run)=='real_pilot_rejected_no_benchmark'
    assert called==['pilot_A0','pilot_A2','pilot_verify']


def test_success_requires_every_serial_phase_and_unknown_outcome_is_rejected():
    m=module();called=[]
    def run(name):called.append(name);return dict(decision=dict(passed=True))
    assert m.execute_sequence(m.ADMITTED,run)=='full_benchmarks_and_report_complete'
    assert called==['pilot_A0','pilot_A2','pilot_verify','freeze','seen_A0','seen_A2','unseen_A0','unseen_A2','report']
    with pytest.raises(ValueError):m.execute_sequence('partial_success',run)


def test_transient_stop_state_is_waited_for_instead_of_promoted_or_restarted():
    m=module()
    assert m.predecessor_outcome(service(MainPID='0',SubState='stop'),record(),identity()) is None


def test_wait_then_rejected_confirmation_never_allocates_gpu_or_loads_candidate(tmp_path,monkeypatch):
    import json
    from experiments.a2_waldo_landmark_belief.paths import save,sha
    m=module();root=tmp_path
    save({},root/'post_pipeline_signature.json');ids=identity();ids['queue']['signature_sha256']=sha(root/'post_pipeline_signature.json')
    save(ids,root/'confirmation_service_identity.json');end=record('VERIFY_rejected_no_forced_enable');end['signature_sha256']=ids['queue']['signature_sha256']
    save(end,root/'post_pipeline_status.json')
    monkeypatch.setattr(m,'require_cpu_gates',lambda root:None);monkeypatch.setattr(m,'cpu_signature',lambda root:{'code':'current'})
    monkeypatch.setattr(m,'launch_identity',lambda:dict(systemd_invocation_id='d'*32,native_main_pid=41,execution_script_sha256='e'*64))
    states=iter([service(),service(MainPID='0',SubState='exited')]);monkeypatch.setattr(m,'service_status',lambda:next(states))
    sleeps=[];monkeypatch.setattr(m.time,'sleep',lambda seconds:sleeps.append(seconds))
    def forbidden(*args,**kwargs):raise AssertionError('Rejected confirmation must not launch inference')
    monkeypatch.setattr(m,'gpu_pids',forbidden);monkeypatch.setattr(m,'load_candidate',forbidden);monkeypatch.setattr(m.subprocess,'run',forbidden)
    m.run_pipeline(root=root,poll_seconds=1)
    result=json.loads((root/'navigation_pipeline_status.json').read_text())
    assert result['state']=='internal_confirmation_not_admitted' and sleeps==[1]
    assert not result['objective_achieved'] and not result['full_benchmark_completed']


def test_terminal_verification_rejects_a_source_snapshot_from_another_queue(tmp_path,monkeypatch):
    from experiments.a2_waldo_landmark_belief.paths import save,sha
    m=module();save({'old_source':1},tmp_path/'navigation_pipeline_signature.json')
    status=dict(state='internal_confirmation_not_admitted',phase='finished',native_main_pid=31,
        systemd_invocation_id='a'*32,execution_script_sha256=sha(m.ROOT/m.WRAPPER),full_benchmark_completed=False,
        signature_sha256=sha(tmp_path/'navigation_pipeline_signature.json'))
    save(status,tmp_path/'navigation_pipeline_status.json')
    monkeypatch.setattr(m,'require_cpu_gates',lambda root:None)
    monkeypatch.setattr(m,'cpu_signature',lambda root:{'new_source':1})
    monkeypatch.setattr(m,'service_status',lambda unit:service(MainPID='0',SubState='exited'))
    with pytest.raises(ValueError,match='snapshot'):m.verify_terminal(tmp_path)
