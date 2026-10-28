import json
from pathlib import Path
import pytest
from experiments.a2_waldo_landmark_belief import relation_v3_navigation_pipeline as pipeline


def service(*,pid=0,state='active',substate='exited',result='success',loaded='loaded',executed=123):
    return dict(LoadState=loaded,MainPID=str(pid),ActiveState=state,SubState=substate,Result=result,ExecMainStatus='0',ExecMainPID=str(executed),ExecStart='reviewed relation_v3_confirmation_pipeline',InvocationID='d'*32)


def test_live_predecessor_cannot_be_replaced_by_a_finished_status_file():
    assert pipeline.predecessor_outcome(service(pid=123,substate='running'),dict(pid=123,phase='finished',state='calibration_passed_real_pilot_pending')) is None


@pytest.mark.parametrize('case',['failed','missing','wrong_pid','missing_file','unknown_state'])
def test_invalid_predecessor_blocks_all_navigation(case):
    info=service();record=dict(pid=123,phase='finished',state='calibration_passed_real_pilot_pending')
    if case=='failed':info['Result']='exit-code'
    elif case=='missing':info['LoadState']='not-found'
    elif case=='wrong_pid':record['pid']=999
    elif case=='missing_file':record=None
    elif case=='unknown_state':record['state']='invented success'
    with pytest.raises(ValueError):pipeline.predecessor_outcome(info,record)


@pytest.mark.parametrize('outcome',['FIT_no_candidate_VERIFY_unconsumed','VERIFY_rejected_no_forced_enable','full_training_no_eligible_DEV_gate_confirmation_unconsumed'])
def test_rejected_internal_results_never_force_navigation(outcome):
    assert pipeline.predecessor_outcome(service(),dict(pid=123,phase='finished',state=outcome))==outcome
    called=[]
    assert pipeline.execute_sequence(outcome,lambda name:called.append(name))=='internal_confirmation_not_admitted'
    assert not called


def test_complete_sequence_is_serial_and_reports_only_after_both_complete_splits():
    active=[];seen=[]
    def run(name):
        assert not active;active.append(name);seen.append(name);active.pop()
        return dict(decision=dict(passed=True)) if name=='pilot_verify' else {}
    assert pipeline.execute_sequence('calibration_passed_real_pilot_pending',run)=='full_benchmarks_and_report_complete'
    assert seen==['pilot_A0','pilot_A2','pilot_verify','freeze','seen_A0','seen_A2','unseen_A0','unseen_A2','report']


def test_failed_real_pilot_cannot_freeze_or_use_formal_validation():
    called=[]
    def run(name):called.append(name);return dict(decision=dict(passed=False))
    assert pipeline.execute_sequence('calibration_passed_real_pilot_pending',run)=='real_pilot_rejected_no_benchmark'
    assert called==['pilot_A0','pilot_A2','pilot_verify']


def test_child_failure_stops_sequence_without_retrying_or_using_unseen():
    called=[]
    def run(name):
        called.append(name)
        if name=='pilot_A2':raise RuntimeError('invented child failure')
    with pytest.raises(RuntimeError):pipeline.execute_sequence('calibration_passed_real_pilot_pending',run)
    assert called==['pilot_A0','pilot_A2']


def test_final_report_requires_complete_bound_files_and_both_full_populations(tmp_path):
    root=tmp_path/'revision_v3';root.mkdir();(root/'frozen.json').write_text('fixed')
    output=tmp_path/'final';output.mkdir();(output/'results.json').write_text(json.dumps(dict(
        status='complete_full_frozen_V3_A0_A2_comparison',freeze_sha256=pipeline.digest(root/'frozen.json'),
        splits={'val_seen':dict(episodes=2470),'val_unseen':dict(episodes=2697)},report_source_sha256={'reviewed':'source'})))
    for name in ('metrics.csv','by_map.csv','ANALYSIS_ZH.md','training_curves.png','training_curves.svg','paired_differences.png','paired_differences.svg','examples.json'):(output/name).write_text('synthetic checked artifact')
    complete=dict(status='complete_full_frozen_V3_A0_A2_comparison',freeze_sha256=pipeline.digest(root/'frozen.json'),files_sha256={p.name:pipeline.digest(p) for p in output.iterdir() if p.is_file()})
    (output/'complete.json').write_text(json.dumps(complete))
    assert pipeline.require_report(output,root,{'reviewed':'source'})['status']==complete['status']
    (output/'results.json').write_text('{}')
    with pytest.raises(ValueError):pipeline.require_report(output,root,{'reviewed':'source'})


def test_report_rejects_path_escape_even_if_bytes_and_hash_match(tmp_path):
    root=tmp_path/'v3';root.mkdir();(root/'frozen.json').write_text('fixed')
    outside=tmp_path/'outside';outside.write_text('private');output=tmp_path/'report';output.mkdir()
    (output/'complete.json').write_text(json.dumps(dict(status='complete_full_frozen_V3_A0_A2_comparison',freeze_sha256=pipeline.digest(root/'frozen.json'),files_sha256={'../outside':pipeline.digest(outside)})))
    with pytest.raises(ValueError):pipeline.require_report(output,root,{})


def test_real_scheduler_waits_for_live_predecessor_then_waits_for_gpu_and_runs_all_children_serially(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from experiments.a2_waldo_landmark_belief import paths,collected_snapshot,relation_v3_navigation_sources as sources
    from experiments.a2_waldo_landmark_belief import relation_v3_pipeline as training,relation_v3_navigation_admission as admission
    monkeypatch.setattr(paths,'ROOT',tmp_path);monkeypatch.setattr(collected_snapshot,'BASE',tmp_path)
    root=tmp_path/'revision_v3';root.mkdir();(root/'post_pipeline_status.json').write_text(json.dumps(dict(pid=123,phase='finished',state=pipeline.ADMITTED)))
    events=[];checks=[]
    current={'bound':'source','guest_wrapper_sha256':'wrapper','navigation':{'relation_v3_confirmation_pipeline.py':'post'}}
    monkeypatch.setattr(pipeline,'signature',lambda:current)
    identity=dict(unit=pipeline.POST_UNIT,module='experiments.a2_waldo_landmark_belief.relation_v3_confirmation_pipeline',source_root=str(tmp_path),execution_host_root=str(tmp_path),wrapper_sha256='wrapper',module_sha256='post',status_sha256=pipeline.digest(root/'post_pipeline_status.json'),guest_pid=123,host_main_pid=123,invocation_id='d'*32,terminal=True)
    (root/'service_identity').mkdir();(root/'service_identity'/(pipeline.POST_UNIT+'.json')).write_text(json.dumps(identity))
    monkeypatch.delenv('A2_EXECUTION_HOST_ROOT',raising=False)
    monkeypatch.setattr(pipeline,'require_cpu_gates',lambda root:checks.append('handoff'))
    monkeypatch.setattr(sources,'require_cpu_gates',lambda root:checks.append('navigation'))
    running=service(pid=123,substate='running');terminal=service()
    for state in (running,terminal):state['ExecStart']=str(tmp_path)+' relation_v3_confirmation_pipeline'
    states=iter([running,terminal])
    monkeypatch.setattr(pipeline,'service_status',lambda:next(states))
    monkeypatch.setattr(pipeline.time,'sleep',lambda seconds:events.append(('wait',seconds)))
    monkeypatch.setattr(admission,'load_candidate',lambda root:events.append(('admit',)))
    gpu=iter([[999],[]]);monkeypatch.setattr(training,'gpu_pids',lambda:next(gpu,[]))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    def child(command,**kwargs):
        matches=[name for name,(module,args,_) in pipeline.PHASES.items() if command[-(len(args)+1):]==['experiments.a2_waldo_landmark_belief.'+module,*args]]
        assert len(matches)==1;name=matches[0]
        assert kwargs['env']['CUDA_VISIBLE_DEVICES']==('0' if pipeline.PHASES[name][2] else '')
        events.append(('child',name));return SimpleNamespace(returncode=0)
    monkeypatch.setattr(pipeline.subprocess,'run',child)
    monkeypatch.setattr(pipeline,'validate_phase',lambda name,root,current:dict(decision=dict(passed=True)) if name=='pilot_verify' else {})
    pipeline.run_pipeline()
    assert events[:3]==[('wait',60),('admit',),('wait',60)]
    assert [event[1] for event in events if event[0]=='child']==list(pipeline.PHASES)
    assert len(checks)>2
    status=json.loads((root/'navigation_pipeline_status.json').read_text())
    assert status['state']=='full_benchmarks_and_report_complete' and status['full_benchmark_completed'] is True


def test_real_phase_proof_rejects_wrong_arm_and_changed_combined_predictions(tmp_path):
    current={'navigation':{'runtime':'fixed'}};root=tmp_path
    directory=root/'evaluation/val_seen/A2';directory.mkdir(parents=True)
    signed=dict(scope='val_seen',arm='A2',code_sha256=current['navigation'])
    (directory/'signature.json').write_text(json.dumps(signed));(directory/'predictions.jsonl').write_text('synthetic verified child output');(directory/'metrics.json').write_text('{}')
    def commit():
        (directory/'complete.json').write_text(json.dumps(dict(status='complete',episodes=2470,signature_sha256=pipeline.digest(directory/'signature.json'),predictions_sha256=pipeline.digest(directory/'predictions.jsonl'),metrics_sha256=pipeline.digest(directory/'metrics.json'))))
    commit();assert pipeline.validate_phase('seen_A2',root,current)['episodes']==2470
    signed['arm']='A0';(directory/'signature.json').write_text(json.dumps(signed));commit()
    with pytest.raises(ValueError):pipeline.validate_phase('seen_A2',root,current)
    signed['arm']='A2';(directory/'signature.json').write_text(json.dumps(signed));commit();(directory/'predictions.jsonl').write_text('mutated')
    with pytest.raises(ValueError):pipeline.validate_phase('seen_A2',root,current)


def test_second_scheduler_cannot_overwrite_first_status_or_enter_runtime(tmp_path,monkeypatch):
    import fcntl
    from experiments.a2_waldo_landmark_belief import collected_snapshot
    monkeypatch.setattr(collected_snapshot,'BASE',tmp_path)
    monkeypatch.setattr(pipeline.sys,'argv',['handoff'])
    root=tmp_path/'revision_v3';root.mkdir();status=root/'navigation_pipeline_status.json';status.write_text('first live supervisor')
    monkeypatch.setattr(pipeline,'run_pipeline',lambda:pytest.fail('second supervisor entered runtime'))
    with (root/'navigation_pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):pipeline.main()
    assert status.read_text()=='first live supervisor'
