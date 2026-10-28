import hashlib,json,sys,types
from pathlib import Path
import pytest
from scripts.a2_remote_bootstrap import mapped_path,archive_ready,archive_untrained_proofs,NAVIGATION_CERTIFICATION


def test_legacy_paths_remain_inside_the_owned_mirror(tmp_path):
    assert mapped_path('/home/20261024_1/Workspace/a',tmp_path)==tmp_path/'legacy_home/Workspace/a'
    for name in ('/etc/passwd','/home/20261024_1/../escape','relative'):
        with pytest.raises(ValueError):mapped_path(name,tmp_path)


def test_partial_archive_waits_but_bad_complete_archive_is_rejected(tmp_path):
    path=tmp_path/'data.tar.zst';expected=dict(bytes=4,sha256=hashlib.sha256(b'good').hexdigest())
    assert not archive_ready(path,expected)
    path.write_bytes(b'go');assert not archive_ready(path,expected)
    path.write_bytes(b'evil')
    with pytest.raises(ValueError):archive_ready(path,expected)
    path.write_bytes(b'good');assert archive_ready(path,expected)


def test_archival_is_restart_safe_and_does_not_remove_the_new_initial(tmp_path):
    (tmp_path/'pretraining').mkdir();(tmp_path/'pretraining/initial.pt').write_bytes(b'old')
    (tmp_path/'cpu_gates.json').write_text(json.dumps(dict(initial_weights_sha256='f'*64)))
    archive=archive_untrained_proofs(tmp_path)
    assert (archive/'pretraining/initial.pt').read_bytes()==b'old'
    assert not (tmp_path/'pretraining').exists()
    (tmp_path/'pretraining').mkdir();(tmp_path/'pretraining/initial.pt').write_bytes(b'new')
    assert archive_untrained_proofs(tmp_path)==archive
    assert (tmp_path/'pretraining/initial.pt').read_bytes()==b'new'


def test_archival_refuses_any_existing_formal_checkpoint(tmp_path):
    (tmp_path/'training').mkdir();(tmp_path/'training/last.pt').write_bytes(b'trained')
    with pytest.raises(ValueError):archive_untrained_proofs(tmp_path)
    assert (tmp_path/'training/last.pt').read_bytes()==b'trained'


def test_archived_old_weights_and_proofs_remain_checked_on_resume(tmp_path):
    (tmp_path/'cpu_gates.json').write_text(json.dumps(dict(initial_weights_sha256='a'*64)))
    archive=archive_untrained_proofs(tmp_path)
    (archive/'cpu_gates.json').write_text('{}')
    with pytest.raises(ValueError):archive_untrained_proofs(tmp_path)


@pytest.mark.parametrize('exit_code,changed,accepted',[(0,False,True),(1,False,False),(0,True,False)])
def test_actual_inline_CPU_certifier_parses_real_summary_and_never_certifies_failed_or_changed_source(tmp_path,monkeypatch,exit_code,changed,accepted):
    import subprocess
    root=tmp_path/'revision_v3';root.mkdir()
    for name in ('navigation_pilot_population.json','navigation_protocol.json'):(root/name).write_text('{}')
    state={'changed':False};admitted=[]
    def save(value,path):Path(path).write_text(json.dumps(value))
    mappings={
        'experiments.a2_waldo_landmark_belief.paths':dict(ROOT=tmp_path,sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest(),save=save),
        'experiments.a2_waldo_landmark_belief.collected_snapshot':dict(BASE=tmp_path),
        'experiments.a2_waldo_landmark_belief.relation_v3_navigation_sources':dict(code_hashes=lambda:{'source':str(state['changed'])},test_hashes=lambda:{'test':'hash'},require_cpu_gates=lambda p:admitted.append(p))}
    for name,attributes in mappings.items():
        module=types.ModuleType(name);module.__dict__.update(attributes);monkeypatch.setitem(sys.modules,name,module)
    def run(command,**kwargs):
        assert command[-3:]==['tests','tests_confirmation_v3','tests_navigation_v3']
        kwargs['stdout'].write('495 passed, 1 warning in 7.0s\n');state['changed']=changed
        return types.SimpleNamespace(returncode=exit_code)
    monkeypatch.setattr(subprocess,'run',run)
    if accepted:
        exec(NAVIGATION_CERTIFICATION,{})
        proof=json.loads((root/'navigation_cpu_gates.json').read_text())
        assert proof['unit_test_summary']=='495 passed, 1 warning in 7.0s' and admitted==[root]
    else:
        with pytest.raises(ValueError):exec(NAVIGATION_CERTIFICATION,{})
        assert not (root/'navigation_cpu_gates.json').exists() and not admitted


@pytest.mark.parametrize('name',['epoch_01.pt','best.pt','history.jsonl','training_complete.json','protocol.json'])
def test_orphan_training_evidence_prevents_recertification_even_after_archival(tmp_path,name):
    archive_untrained_proofs(tmp_path)
    training=tmp_path/'training';training.mkdir()
    evidence=training/name;evidence.write_bytes(b'existing formal record')
    with pytest.raises(ValueError,match='formal training'):
        archive_untrained_proofs(tmp_path)
    assert evidence.read_bytes()==b'existing formal record'


def test_singleton_lock_prevents_second_preparation_without_touching_status(tmp_path):
    from scripts.a2_remote_bootstrap import host_lock
    status=tmp_path/'bootstrap_status.json';status.write_text('first process')
    with host_lock(tmp_path):
        with pytest.raises(BlockingIOError):
            with host_lock(tmp_path):pytest.fail('second owner acquired lock')
        assert status.read_text()=='first process'
    with host_lock(tmp_path):pass


@pytest.mark.parametrize('target',[None,'local','remote'])
def test_only_explicit_remote_target_arms_training(tmp_path,monkeypatch,target):
    from scripts.a2_remote_bootstrap import Bootstrap
    job=Bootstrap(tmp_path);started=[]
    if target is not None:(tmp_path/'execution_target.json').write_text(json.dumps({'target':target}))
    for name in ('verify_launch','wait_extract','verify_inputs','recertify'):
        monkeypatch.setattr(job,name,lambda:None)
    monkeypatch.setattr(job,'start',lambda *args:started.append(args))
    monkeypatch.setattr(job,'guest',lambda *args,**kwargs:None)
    monkeypatch.setattr(job,'service',lambda name:dict(ActiveState='inactive',Result='success',ExecMainStatus='0',MainPID='0',SubState='dead'))
    job.science.mkdir(parents=True);(job.science/'post_pipeline_status.json').write_text('{"state":"finished"}')
    (job.science/'navigation_pipeline_status.json').write_text('{"state":"finished","full_benchmark_completed":false}')
    job.run()
    assert len(started)==(3 if target=='remote' else 0)


def test_launch_binding_rejects_wrong_revision_source_or_manifest(tmp_path,monkeypatch):
    from scripts.a2_remote_bootstrap import Bootstrap,digest,INPUT_MANIFESTS,POLICY
    job=Bootstrap(tmp_path);job.project.mkdir(parents=True)
    source=job.project/'model.py';source.write_text('reviewed model')
    active=job.project/POLICY;active.parent.mkdir(parents=True);active.write_text('{}')
    manifest=tmp_path/'reuse_manifest.json'
    for name in INPUT_MANIFESTS:(tmp_path/name).write_text('{}')
    launch=tmp_path/'launch_manifest.json'
    launch.write_text(json.dumps(dict(code_commit='reviewed',branch='a2',source_sha256={'model.py':digest(source)},input_sha256={name:digest(tmp_path/name) for name in INPUT_MANIFESTS})))
    job.launch_sha256=digest(launch)
    state={'head':'reviewed','branch':'a2'}
    def output(command,**kwargs):
        if command[1:]==['rev-parse','HEAD']:return state['head']+'\n'
        if command[1:]==['branch','--show-current']:return state['branch']+'\n'
        if command[1:]==['ls-files','-z']:return 'model.py\0'
        raise AssertionError(command)
    monkeypatch.setattr('scripts.a2_remote_bootstrap.subprocess.check_output',output)
    # Avoid the unrelated running bootstrap path assertion for this synthetic checkout.
    monkeypatch.setattr('scripts.a2_remote_bootstrap.__file__',str(source))
    job.verify_launch()
    started=[]
    (tmp_path/'execution_target.json').write_text('{"target":"remote"}')
    monkeypatch.setattr(job,'wait_extract',lambda:None)
    monkeypatch.setattr(job,'verify_inputs',lambda:None)
    monkeypatch.setattr(job,'recertify',lambda:active.write_text('{"changed":true}'))
    monkeypatch.setattr(job,'start',lambda *args:started.append(args))
    monkeypatch.setattr(job,'guest',lambda *args,**kwargs:None)
    with pytest.raises(ValueError,match='Installed resource policy'):job.run()
    assert not started
    active.write_text('{}')
    state['head']='unreviewed'
    with pytest.raises(ValueError):job.verify_launch()
    state['head']='reviewed';source.write_text('changed model')
    with pytest.raises(ValueError):job.verify_launch()
    source.write_text('reviewed model');manifest.write_text('{"changed":true}')
    with pytest.raises(ValueError):job.verify_launch()
    manifest.write_text('{}');active.write_text('{"changed":true}')
    with pytest.raises(ValueError):job.verify_launch()
    active.write_text('{}');launch.write_text('{}')
    with pytest.raises(ValueError):job.verify_launch()


def test_changed_source_after_recertification_never_starts_training(tmp_path,monkeypatch):
    from scripts.a2_remote_bootstrap import Bootstrap
    job=Bootstrap(tmp_path);state={'changed':False};started=[]
    (tmp_path/'execution_target.json').write_text('{"target":"remote"}')
    def verify():
        if state['changed']:raise ValueError('reviewed source changed')
    monkeypatch.setattr(job,'verify_launch',verify)
    for name in ('wait_extract','verify_inputs'):monkeypatch.setattr(job,name,lambda:None)
    monkeypatch.setattr(job,'recertify',lambda:state.update(changed=True))
    monkeypatch.setattr(job,'start',lambda *args:started.append(args))
    monkeypatch.setattr(job,'guest',lambda *args,**kwargs:None)
    with pytest.raises(ValueError):job.run()
    assert not started


def test_second_bootstrap_cannot_enter_preparation_or_overwrite_first_status(tmp_path,monkeypatch):
    from scripts.a2_remote_bootstrap import Bootstrap,host_lock
    job=Bootstrap(tmp_path)
    marker=tmp_path/'bootstrap_status.json';marker.write_text('live first owner')
    monkeypatch.setattr(job,'_run',lambda:pytest.fail('second bootstrap entered preparation'))
    with host_lock(tmp_path):
        with pytest.raises(BlockingIOError):job.run()
    assert marker.read_text()=='live first owner'
