import json,os
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts import a2_guest_service as wrapper
from experiments.a2_waldo_landmark_belief import relation_v3_navigation_pipeline as pipeline


def test_wrapper_commits_actual_guest_pid_and_successful_status_without_modifying_scientific_module(tmp_path,monkeypatch):
    project=tmp_path;script=project/'scripts/a2_guest_service.py';script.parent.mkdir();script.write_text('reviewed wrapper')
    target=project/'experiments/a2_waldo_landmark_belief/relation_v3_confirmation_pipeline.py';target.parent.mkdir(parents=True);target.write_text('reviewed scientific module')
    monkeypatch.setattr(wrapper,'__file__',str(script))
    unit=pipeline.POST_UNIT;module='experiments.a2_waldo_landmark_belief.relation_v3_confirmation_pipeline'
    identity=dict(unit=unit,invocation_id='d'*32,host_main_pid=100,guest_pid=os.getpid(),execution_host_root=str(project))
    monkeypatch.setattr(wrapper,'current_service',lambda unit:identity)
    status_path=project/'runs/a2_waldo_landmark_20261027/revision_v3/post_pipeline_status.json'
    called=[]
    def body(name,**kwargs):
        called.append(name);status_path.write_text(json.dumps(dict(pid=os.getpid(),phase='finished',state=pipeline.ADMITTED)))
    monkeypatch.setattr(wrapper.runpy,'run_module',body)
    wrapper.run(unit,module)
    saved=json.loads((status_path.parent/'service_identity'/(unit+'.json')).read_text())
    assert saved['guest_pid']!=saved['host_main_pid'] and saved['terminal'] is True
    assert saved['status_sha256']==wrapper.digest(status_path) and called==[module]
    service=dict(ExecMainPID='100',InvocationID='d'*32)
    assert wrapper.require_identity(service,json.loads(status_path.read_text()),saved,{'module':module})==saved
    service['InvocationID']='e'*32
    with pytest.raises(ValueError):wrapper.require_identity(service,json.loads(status_path.read_text()),saved,{'module':module})


def test_current_service_requires_actual_matching_systemd_invocation_and_host_root(monkeypatch):
    monkeypatch.setenv('INVOCATION_ID','d'*32);monkeypatch.setenv('A2_EXECUTION_HOST_ROOT','/owned/task')
    reply='MainPID=100\nInvocationID='+('d'*32)+'\nExecStart=/owned/task/tools/proot wrapper\n'
    monkeypatch.setattr(wrapper.subprocess,'check_output',lambda *args,**kwargs:reply)
    assert wrapper.current_service('owned.service')['host_main_pid']==100
    monkeypatch.setenv('INVOCATION_ID','e'*32)
    with pytest.raises(ValueError):wrapper.current_service('owned.service')


def terminal_runtime(tmp_path,monkeypatch):
    from experiments.a2_waldo_landmark_belief import paths,collected_snapshot,relation_v3_navigation_sources as sources,relation_v3_navigation_freeze as freeze
    monkeypatch.setattr(paths,'ROOT',tmp_path);monkeypatch.setattr(collected_snapshot,'BASE',tmp_path/'run');monkeypatch.delenv('A2_EXECUTION_HOST_ROOT',raising=False)
    root=tmp_path/'run/revision_v3';root.mkdir(parents=True)
    record=dict(pid=999,phase='finished',state='full_benchmarks_and_report_complete',full_benchmark_completed=True)
    status=root/'navigation_pipeline_status.json';status.write_text(json.dumps(record))
    identity=dict(unit=pipeline.NAV_UNIT,invocation_id='d'*32,host_main_pid=100,guest_pid=999,terminal=True,
        source_root=str(tmp_path),execution_host_root=str(tmp_path),module='experiments.a2_waldo_landmark_belief.relation_v3_navigation_pipeline',
        wrapper_sha256='wrapper',module_sha256='pipeline',status_sha256=pipeline.digest(status))
    identity_path=root/'service_identity'/(pipeline.NAV_UNIT+'.json');identity_path.parent.mkdir();identity_path.write_text(json.dumps(identity))
    service=dict(LoadState='loaded',MainPID='0',ExecMainPID='100',InvocationID='d'*32,Result='success',ExecMainStatus='0',ActiveState='active',SubState='exited')
    current=dict(guest_wrapper_sha256='wrapper',pipeline_sha256='pipeline',reporting={'report':'source'})
    monkeypatch.setattr(pipeline,'signature',lambda:current);monkeypatch.setattr(pipeline,'require_cpu_gates',lambda root:None)
    monkeypatch.setattr(sources,'require_cpu_gates',lambda root:None);monkeypatch.setattr(pipeline,'service_status',lambda unit:service)
    checks=[];monkeypatch.setattr(freeze,'validate_frozen',lambda root:checks.append('frozen'))
    return root,record,identity,identity_path,checks


def test_full_completion_requires_current_frozen_report_even_with_correct_wrapped_execution(tmp_path,monkeypatch):
    root,record,identity,identity_path,checks=terminal_runtime(tmp_path,monkeypatch)
    with pytest.raises(FileNotFoundError):pipeline.verify_terminal()
    assert checks==['frozen']
    monkeypatch.setattr(pipeline,'require_report',lambda *args:checks.append('current_report'))
    assert pipeline.verify_terminal()==record and checks[-2:]==['frozen','current_report']


@pytest.mark.parametrize('mutation',['phase','pid','invocation','status_sha'])
def test_stale_terminal_navigation_never_reaches_report_completion(tmp_path,monkeypatch,mutation):
    root,record,identity,identity_path,checks=terminal_runtime(tmp_path,monkeypatch)
    if mutation=='phase':record['phase']='not_finished'
    elif mutation=='pid':record['pid']=998
    elif mutation=='invocation':identity['invocation_id']='e'*32
    elif mutation=='status_sha':identity['status_sha256']='stale'
    (root/'navigation_pipeline_status.json').write_text(json.dumps(record));identity_path.write_text(json.dumps(identity))
    with pytest.raises(ValueError):pipeline.verify_terminal()
    assert not checks


def test_actual_bootstrap_rejects_stale_true_status_instead_of_claiming_full_completion(tmp_path,monkeypatch):
    from scripts.a2_remote_bootstrap import Bootstrap
    root,record,identity,identity_path,checks=terminal_runtime(tmp_path,monkeypatch)
    record['phase']='not_finished';(root/'navigation_pipeline_status.json').write_text(json.dumps(record))
    (root/'post_pipeline_status.json').write_text('{"state":"finished"}')
    job=Bootstrap(tmp_path);job.science=root
    (tmp_path/'execution_target.json').write_text('{"target":"remote"}')
    for name in ('verify_launch','wait_extract','verify_inputs','recertify'):monkeypatch.setattr(job,name,lambda:None)
    started=[];monkeypatch.setattr(job,'start',lambda *args:started.append(args))
    monkeypatch.setattr(job,'service',lambda unit:dict(ActiveState='active',SubState='exited',Result='success',ExecMainStatus='0',MainPID='0'))
    def guest(args,label,**kwargs):
        assert args[-1]=='--verify-complete' and label=='verify_terminal_navigation'
        pipeline.verify_terminal()
    monkeypatch.setattr(job,'guest',guest)
    with pytest.raises(ValueError):job.run()
    status=json.loads((tmp_path/'bootstrap_status.json').read_text())
    assert status['state']=='failed' and not status.get('full_benchmark_completed',False)
    assert len(started)==3
