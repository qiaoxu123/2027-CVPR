"""Serial post-confirmation real pilot, freeze, complete benchmarks and report.

A live predecessor always wins over a status file. Rejected internal evidence
never enables correction. No checkpoint/threshold or episode-limit option exists.
"""
import argparse,fcntl,hashlib,json,os,re,subprocess,sys,time
from pathlib import Path

POST_UNIT='graphnav-a2-v3-confirmation-20261028.service'
NAV_UNIT='graphnav-a2-v3-navigation-20261028.service'
ADMITTED='calibration_passed_real_pilot_pending'
REJECTED={'FIT_no_candidate_VERIFY_unconsumed','VERIFY_rejected_no_forced_enable','full_training_no_eligible_DEV_gate_confirmation_unconsumed'}
PHASES=dict(
    pilot_A0=('relation_v3_navigation',('--scope','pilot','--arm','A0'),True),
    pilot_A2=('relation_v3_navigation',('--scope','pilot','--arm','A2'),True),
    pilot_verify=('relation_v3_navigation',('--verify-pilot',),False),
    freeze=('relation_v3_navigation_freeze',(),False),
    seen_A0=('relation_v3_navigation',('--scope','val_seen','--arm','A0'),True),
    seen_A2=('relation_v3_navigation',('--scope','val_seen','--arm','A2'),True),
    unseen_A0=('relation_v3_navigation',('--scope','val_unseen','--arm','A0'),True),
    unseen_A2=('relation_v3_navigation',('--scope','val_unseen','--arm','A2'),True),
    report=('relation_v3_results',(),False))
COUNTS=dict(val_seen=2470,val_unseen=2697)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()


def predecessor_outcome(service,record,identity=None,expected=None):
    if (service['LoadState']!='loaded' or service['ActiveState']=='failed' or service['Result']!='success' or int(service['ExecMainStatus'])!=0):
        raise ValueError('Confirmation predecessor missing or failed; inspect rather than restart')
    if int(service['MainPID'])>0 or service['SubState'] in ('running','start','start-pre','start-post','stop','stop-sigterm','stop-sigkill'):return None
    if (service['ActiveState'],service['SubState']) not in (('active','exited'),('inactive','dead')):raise ValueError('Unknown predecessor lifecycle')
    if record is None or record.get('phase')!='finished':raise ValueError('Terminal confirmation lacks its own complete status')
    if expected is not None:
        from scripts.a2_guest_service import require_identity
        require_identity(service,record,identity,expected)
    elif record.get('pid')!=int(service['ExecMainPID']) or int(service['ExecMainPID'])<=0:raise ValueError('Terminal confirmation lacks its own complete status')
    outcome=record.get('state')
    if outcome not in REJECTED|{ADMITTED}:raise ValueError('Unknown confirmation outcome')
    return outcome


def execute_sequence(outcome,run_phase):
    if outcome in REJECTED:return 'internal_confirmation_not_admitted'
    if outcome!=ADMITTED:raise ValueError('Unadmitted confirmation outcome')
    for name in PHASES:
        result=run_phase(name)
        if name=='pilot_verify' and not result['decision']['passed']:return 'real_pilot_rejected_no_benchmark'
    return 'full_benchmarks_and_report_complete'


def require_report(output,root,report_sources):
    output=Path(output);root=Path(root);complete=json.loads((output/'complete.json').read_text())
    expected='complete_full_frozen_V3_A0_A2_comparison'
    if complete['status']!=expected or complete['freeze_sha256']!=digest(root/'frozen.json'):raise ValueError('Final report freeze/status changed')
    bound=complete['files_sha256']
    required={'results.json','metrics.csv','by_map.csv','ANALYSIS_ZH.md','training_curves.png','training_curves.svg','paired_differences.png','paired_differences.svg','examples.json'}
    if not required<=set(bound):raise ValueError('Final report lacks required artifact bindings')
    for name,expected_sha in bound.items():
        relative=Path(name);path=output/relative
        if relative.is_absolute() or '..' in relative.parts or path.resolve().parent!=output.resolve():raise ValueError('Unsafe report artifact path')
        if digest(path)!=expected_sha:raise ValueError('Completed report artifact changed')
    result=json.loads((output/'results.json').read_text())
    if (result['status']!=expected or result['freeze_sha256']!=complete['freeze_sha256'] or result['report_source_sha256']!=report_sources
        or set(result['splits'])!=set(COUNTS) or any(result['splits'][scope]['episodes']!=n for scope,n in COUNTS.items())):raise ValueError('Full frozen paired report required')
    return complete


def signature():
    from .paths import ROOT
    from .relation_v3_navigation_sources import code_hashes,test_hashes
    from .relation_v3_preparation import test_hashes as train_tests
    from .relation_v3_confirmation_pipeline import post_test_hashes
    from .relation_v3_results import reporting_hashes
    own=sorted((ROOT/'tests_handoff_v3').rglob('*.py'))
    if not own:raise ValueError('Navigation handoff test inventory missing')
    return dict(navigation=code_hashes(),navigation_tests=test_hashes(),training_tests=train_tests(),confirmation_tests=post_test_hashes(),
        reporting=reporting_hashes(ROOT),handoff={str(p.relative_to(ROOT)):digest(p) for p in own},pipeline_sha256=digest(__file__),guest_wrapper_sha256=digest(ROOT/'scripts/a2_guest_service.py'),
        remote_preparation={str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'scripts/a2_remote_bootstrap.py']+sorted((ROOT/'tests_remote_migration_v3').rglob('*.py'))})


def require_cpu_gates(root):
    root=Path(root);proof=json.loads((root/'navigation_pipeline_cpu_gates.json').read_text())
    if (proof['status']!='v3_navigation_handoff_CPU_gates_passed' or proof['implementation']!=signature()
        or proof['test_log_sha256']!=digest(root/'navigation_pipeline_cpu_tests.log')):raise ValueError('Current navigation handoff CPU proof required')
    return proof


def cpu_gates():
    from .paths import ROOT,save
    from .collected_snapshot import BASE
    root=BASE/'revision_v3';root.mkdir(parents=True,exist_ok=True);before=signature();log=root/'navigation_pipeline_cpu_tests.log'
    proof=root/'navigation_pipeline_cpu_gates.json'
    if proof.exists():require_cpu_gates(root);return
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    directories=['tests','tests_confirmation_v3','tests_navigation_v3','tests_remote_migration_v3','tests_reporting_v3','tests_handoff_v3']
    with log.open('w') as stream:result=subprocess.run([sys.executable,'-m','pytest','-q',*directories],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    match=re.search(r'\d+ passed[^\n]*',log.read_text())
    if result.returncode or match is None or signature()!=before:raise ValueError('Full handoff CPU check failed or source changed')
    save(dict(status='v3_navigation_handoff_CPU_gates_passed',implementation=before,test_log_sha256=digest(log),unit_test_summary=match.group(),no_GPU_used=True),proof)
    require_cpu_gates(root)


def service_status(unit=POST_UNIT):
    environment=dict(os.environ,XDG_RUNTIME_DIR=f'/run/user/{os.getuid()}')
    command=['systemctl','--user','show',unit]
    for key in ('LoadState','ActiveState','SubState','MainPID','ExecMainPID','ExecMainStatus','Result','ExecStart','InvocationID'):command+=['-p',key]
    result=subprocess.run(command,env=environment,text=True,capture_output=True)
    parsed=dict(line.split('=',1) for line in result.stdout.strip().splitlines())
    if result.returncode and parsed.get('LoadState')!='not-found':raise RuntimeError('Cannot inspect confirmation predecessor: '+result.stderr)
    return parsed


def execution_identity(unit,root,current):
    from .paths import ROOT
    is_post=unit==POST_UNIT
    if unit not in (POST_UNIT,NAV_UNIT):raise ValueError('Unsupported predecessor identity')
    module='relation_v3_confirmation_pipeline' if is_post else 'relation_v3_navigation_pipeline'
    status_path=root/('post_pipeline_status.json' if is_post else 'navigation_pipeline_status.json')
    identity_path=root/'service_identity'/(unit+'.json')
    identity=json.loads(identity_path.read_text()) if identity_path.exists() else None
    expected=dict(unit=unit,module='experiments.a2_waldo_landmark_belief.'+module,source_root=str(ROOT),
        execution_host_root=os.environ.get('A2_EXECUTION_HOST_ROOT',str(ROOT)),wrapper_sha256=current['guest_wrapper_sha256'],
        module_sha256=current['navigation'][module+'.py'] if is_post else current['pipeline_sha256'],status_sha256=digest(status_path))
    return identity,expected


def verify_terminal():
    from .paths import ROOT
    from .collected_snapshot import BASE
    from scripts.a2_guest_service import require_identity
    from .relation_v3_navigation_sources import require_cpu_gates as navigation_cpu
    from .relation_v3_navigation_freeze import validate_frozen
    root=BASE/'revision_v3';require_cpu_gates(root);navigation_cpu(root);current=signature();service=service_status(NAV_UNIT)
    if (service['LoadState']!='loaded' or int(service['MainPID'])!=0 or service['Result']!='success'
        or int(service['ExecMainStatus'])!=0 or (service['ActiveState'],service['SubState']) not in (('active','exited'),('inactive','dead'))):raise ValueError('Actual navigation service has not completed successfully')
    record=json.loads((root/'navigation_pipeline_status.json').read_text());identity,expected=execution_identity(NAV_UNIT,root,current)
    require_identity(service,record,identity,expected)
    finished=record.get('state')=='full_benchmarks_and_report_complete'
    if record.get('state') not in ('full_benchmarks_and_report_complete','internal_confirmation_not_admitted','real_pilot_rejected_no_benchmark') or record.get('full_benchmark_completed') is not finished:raise ValueError('Unknown or inconsistent final navigation outcome')
    if finished:
        validate_frozen(root);require_report(ROOT/'reports/a2_execution_20261027/revision_v3_final',root,current['reporting'])
    if signature()!=current:raise ValueError('Terminal navigation source changed during verification')
    return record


def completion_path(name,root):
    if name.startswith('pilot_') and name!='pilot_verify':return root/'pilot'/name.split('_',1)[1]/'complete.json'
    if name.startswith(('seen_','unseen_')):
        scope,arm=name.split('_',1);return root/'evaluation'/('val_'+scope)/arm/'complete.json'
    return None


def validate_phase(name,root,current):
    from .paths import ROOT
    path=completion_path(name,root)
    if path is not None:
        complete=json.loads(path.read_text());directory=path.parent;signed=json.loads((directory/'signature.json').read_text())
        prefix,arm=name.split('_');scope='pilot' if prefix=='pilot' else 'val_'+prefix
        count=64 if scope=='pilot' else COUNTS[scope]
        if (signed['scope']!=scope or signed['arm']!=arm or complete['status']!='complete' or complete['episodes']!=count or signed['code_sha256']!=current['navigation']
            or complete['signature_sha256']!=digest(directory/'signature.json') or complete['predictions_sha256']!=digest(directory/'predictions.jsonl')
            or complete['metrics_sha256']!=digest(directory/'metrics.json')):raise ValueError('Navigation child lacks a complete current phase proof')
        return complete
    if name=='pilot_verify':
        report=json.loads((root/'pilot/report.json').read_text())
        if report['code_sha256']!=current['navigation'] or report['decision']['episodes']!=64 or report['benchmark_used']:raise ValueError('Pilot verification lacks its complete current proof')
        return report
    if name=='freeze':
        from .relation_v3_navigation_freeze import validate_frozen
        return validate_frozen(root)
    if name=='report':return require_report(ROOT/'reports/a2_execution_20261027/revision_v3_final',root,current['reporting'])
    raise ValueError('Unknown navigation phase')


def run_pipeline():
    from .paths import ROOT,save
    from .collected_snapshot import BASE
    from .relation_v3_navigation_sources import require_cpu_gates as require_navigation_cpu
    from .relation_v3_pipeline import gpu_pids,is_busy_rejection
    from .relation_v3_navigation_admission import load_candidate
    root=BASE/'revision_v3';require_cpu_gates(root);require_navigation_cpu(root);current=signature()
    logs=root/'navigation_pipeline_logs';logs.mkdir(exist_ok=True)
    def unchanged():
        if signature()!=current:raise ValueError('Queued navigation/report source changed')
        require_cpu_gates(root);require_navigation_cpu(root)
    def status(state,phase,**extra):
        save(dict(state=state,phase=phase,pid=os.getpid(),time=time.time(),**extra),root/'navigation_pipeline_status.json')
    while True:
        unchanged();service=service_status()
        command=service.get('ExecStart','');execution_root=os.environ.get('A2_EXECUTION_HOST_ROOT',str(ROOT))
        if 'relation_v3_confirmation_pipeline' not in command or execution_root not in command:raise ValueError('Predecessor unit belongs to another execution')
        path=root/'post_pipeline_status.json';record=json.loads(path.read_text()) if path.exists() else None
        identity,expected=execution_identity(POST_UNIT,root,current) if int(service['MainPID'])==0 and record is not None else (None,None)
        outcome=predecessor_outcome(service,record,identity,expected)
        if outcome is not None:break
        status('waiting_for_complete_confirmation','confirmation',predecessor_pid=int(service['MainPID']));time.sleep(60)
    if outcome==ADMITTED:load_candidate(root)
    def run_phase(name):
        module,args,needs_gpu=PHASES[name]
        while True:
            unchanged();path=completion_path(name,root)
            if needs_gpu and not (path is not None and path.exists()):
                while True:
                    pids=gpu_pids()
                    if not pids:break
                    status('waiting_for_GPU_idle',name,external_gpu_pids=pids);time.sleep(60);unchanged()
            command=[sys.executable,'-u','-m','experiments.a2_waldo_landmark_belief.'+module,*args]
            environment=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            if not needs_gpu:environment['CUDA_VISIBLE_DEVICES']=''
            log=logs/(name+'.log');offset=log.stat().st_size if log.exists() else 0;status('running',name,log=str(log))
            with log.open('a') as stream:result=subprocess.run(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
            unchanged()
            if result.returncode:
                if needs_gpu and is_busy_rejection(log,offset):status('GPU_became_busy',name);continue
                raise RuntimeError(f'Navigation child failed: {name}, exit={result.returncode}; inspect preserved log')
            result=validate_phase(name,root,current);status('completed_phase',name);return result
    final=execute_sequence(outcome,run_phase)
    status(final,'finished',confirmation_outcome=outcome,full_benchmark_completed=final=='full_benchmarks_and_report_complete')


def main():
    from .paths import save
    from .collected_snapshot import BASE
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group();mode.add_argument('--cpu-gates',action='store_true');mode.add_argument('--verify-complete',action='store_true');args=parser.parse_args()
    root=BASE/'revision_v3';root.mkdir(parents=True,exist_ok=True)
    if args.cpu_gates:cpu_gates();return
    if args.verify_complete:print(json.dumps(verify_terminal()),flush=True);return
    with (root/'navigation_pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run_pipeline()
        except Exception as error:
            save(dict(state='failed',pid=os.getpid(),time=time.time(),reason=str(error)),root/'navigation_pipeline_status.json');raise


if __name__=='__main__':main()
