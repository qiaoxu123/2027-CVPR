"""Serial V4 pilot/freeze/full benchmarks, bound to actual native service identity."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from .paths import ROOT,sha,save
from .reference_overlay_context import BASE
from .reference_v4_confirmation_protocol import immutable
from .reference_v4_confirmation_pipeline import ObservationUnavailable
from .reference_v4_navigation_sources import cpu_signature,require_cpu_gates,code_hashes
from .reference_v4_navigation_admission import load_candidate
from .reference_v4_results import reporting_hashes
from .relation_v3_pipeline import gpu_pids,is_busy_rejection

POST_UNIT='graphnav-a2-reference-v4-confirmation-20261029.service'
NAV_UNIT='graphnav-a2-reference-v4-navigation-20261029.service'
WRAPPER='scripts/a2_reference_v4_remote_navigation.sh'
ADMITTED='calibration_passed_real_pilot_pending'
REJECTED={'FIT_no_candidate_VERIFY_unconsumed','VERIFY_rejected_no_forced_enable','full_training_no_eligible_DEV_gate_confirmation_unconsumed'}
COUNTS=dict(val_seen=2470,val_unseen=2697)
PHASES=dict(pilot_A0=('reference_v4_navigation',('--scope','pilot','--arm','A0'),True),
    pilot_A2=('reference_v4_navigation',('--scope','pilot','--arm','A2'),True),
    pilot_verify=('reference_v4_navigation',('--verify-pilot',),False),
    freeze=('reference_v4_navigation_freeze',(),False),
    seen_A0=('reference_v4_navigation',('--scope','val_seen','--arm','A0'),True),
    seen_A2=('reference_v4_navigation',('--scope','val_seen','--arm','A2'),True),
    unseen_A0=('reference_v4_navigation',('--scope','val_unseen','--arm','A0'),True),
    unseen_A2=('reference_v4_navigation',('--scope','val_unseen','--arm','A2'),True),
    report=('reference_v4_results',(),False))


def service_status(unit=POST_UNIT):
    if unit not in (POST_UNIT,NAV_UNIT):raise ValueError('Unknown V4 unit')
    fields=('LoadState','ActiveState','SubState','MainPID','ExecMainPID','ExecMainStatus','Result','InvocationID')
    command=['systemctl','--user','show',unit]
    for field in fields:command.extend(('-p',field))
    try:
        result=subprocess.run(command,env=dict(os.environ,XDG_RUNTIME_DIR=f'/run/user/{os.getuid()}'),text=True,capture_output=True,timeout=20)
    except (OSError,subprocess.TimeoutExpired) as error:raise ObservationUnavailable(str(error)) from error
    value=dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
    if value.get('LoadState')=='not-found':return value
    if result.returncode or any(field not in value for field in fields):raise ObservationUnavailable(result.stderr.strip())
    return value


def predecessor_outcome(service,record,identity):
    if (service['LoadState']!='loaded' or service['ActiveState']=='failed'
        or service['Result']!='success' or int(service['ExecMainStatus'])!=0):
        raise ValueError('Confirmation predecessor missing or failed; inspect the actual job')
    native=identity['units'][POST_UNIT];guest=identity['queue']
    if service['InvocationID']!=native['InvocationID']:
        raise ValueError('Confirmation unit is another execution instance')
    if int(service['MainPID'])>0:
        if service['MainPID']!=native['MainPID']:raise ValueError('Confirmation native process changed')
        return None
    if service['SubState'] in ('start','start-pre','start-post','stop','stop-sigterm','stop-sigkill'):
        return None
    if (service['ActiveState'],service['SubState']) not in (('active','exited'),('inactive','dead')):
        raise ValueError('Unknown confirmation lifecycle')
    if (str(service['ExecMainPID'])!=str(native['MainPID']) or record is None
        or record.get('pid')!=guest['pid'] or record.get('signature_sha256')!=guest['signature_sha256']
        or record.get('phase')!='finished'):
        raise ValueError('Terminal confirmation does not match its verified native/guest execution')
    if record.get('state') not in REJECTED|{ADMITTED}:raise ValueError('Unknown complete confirmation outcome')
    return record['state']


def execute_sequence(outcome,run_phase):
    if outcome in REJECTED:return 'internal_confirmation_not_admitted'
    if outcome!=ADMITTED:raise ValueError('Unadmitted V4 confirmation outcome')
    for phase in PHASES:
        result=run_phase(phase)
        if phase=='pilot_verify' and not result['decision']['passed']:return 'real_pilot_rejected_no_benchmark'
    return 'full_benchmarks_and_report_complete'


def completion_path(name,root):
    if name.startswith('pilot_') and name!='pilot_verify':return root/'pilot'/name.split('_',1)[1]/'complete.json'
    if name.startswith(('seen_','unseen_')):
        scope,arm=name.split('_',1);return root/'evaluation'/('val_'+scope)/arm/'complete.json'
    return None


def require_report(output,root):
    output=Path(output);complete=json.loads((output/'complete.json').read_text());status='complete_full_frozen_V4_A0_A2_comparison'
    if complete['status']!=status or complete['freeze_sha256']!=sha(root/'frozen.json'):raise ValueError('Report freeze/status changed')
    required={'results.json','metrics.csv','by_map.csv','ANALYSIS_ZH.md','training_curves.png','training_curves.svg','paired_differences.png','paired_differences.svg','examples.json'}
    if not required<=set(complete['files_sha256']):raise ValueError('Report lacks required artifacts')
    for name,digest in complete['files_sha256'].items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or (output/p).resolve().parent!=output.resolve() or sha(output/p)!=digest:
            raise ValueError('Report artifact source changed')
    result=json.loads((output/'results.json').read_text())
    if (result['status']!=status or result['freeze_sha256']!=complete['freeze_sha256']
        or result['navigation_code_sha256']!=code_hashes() or result['report_source_sha256']!=reporting_hashes(ROOT)
        or set(result['splits'])!=set(COUNTS) or any(result['splits'][scope]['episodes']!=n for scope,n in COUNTS.items())):
        raise ValueError('Complete frozen V4 paired report required')
    return complete


def validate_phase(name,root):
    path=completion_path(name,root)
    if path is not None:
        complete=json.loads(path.read_text());directory=path.parent;signed=json.loads((directory/'signature.json').read_text())
        prefix,arm=name.split('_');scope='pilot' if prefix=='pilot' else 'val_'+prefix;count=64 if scope=='pilot' else COUNTS[scope]
        if (signed['scope']!=scope or signed['arm']!=arm or signed['code_sha256']!=code_hashes()
            or complete!=dict(status='complete',episodes=count,signature_sha256=sha(directory/'signature.json'),
                             predictions_sha256=sha(directory/'predictions.jsonl'),metrics_sha256=sha(directory/'metrics.json'))):
            raise ValueError('Navigation child lacks a complete current phase proof')
        return complete
    if name=='pilot_verify':
        report=json.loads((root/'pilot/report.json').read_text())
        if report['code_sha256']!=code_hashes() or report['decision']['episodes']!=64 or report['benchmark_used']:
            raise ValueError('Complete current pilot verification required')
        return report
    if name=='freeze':
        from .reference_v4_navigation_freeze import validate_frozen
        return validate_frozen(root)
    if name=='report':return require_report(ROOT/'reports/a2_execution_20261027/revision_v4_final',root)
    raise ValueError('Unknown V4 navigation phase')


def launch_identity():
    invocation=os.environ.get('INVOCATION_ID','');native=int(os.environ.get('A2_NATIVE_MAIN_PID','0'))
    wrapper=os.environ.get('A2_NAV_WRAPPER_SHA256','')
    if not re.fullmatch('[0-9a-f]{32}',invocation) or native!=os.getppid() or wrapper!=sha(ROOT/WRAPPER):
        raise ValueError('Verified native V4 navigation wrapper required')
    return dict(systemd_invocation_id=invocation,native_main_pid=native,execution_script_sha256=wrapper)


def run_pipeline(*,root=None,poll_seconds=300):
    root=Path(root) if root is not None else BASE/'revision_v4'
    if poll_seconds<=0:raise ValueError('Positive polling interval required')
    require_cpu_gates(root);launch=launch_identity();current=cpu_signature(root)
    immutable(current,root/'navigation_pipeline_signature.json')
    identity=json.loads((root/'confirmation_service_identity.json').read_text())
    if identity['queue']['signature_sha256']!=sha(root/'post_pipeline_signature.json'):
        raise ValueError('Original queued confirmation signature changed')
    logs=root/'navigation_pipeline_logs';logs.mkdir(exist_ok=True)
    def unchanged():
        if cpu_signature(root)!=current:raise ValueError('Queued navigation/report source changed')
        require_cpu_gates(root)
    def status(state,phase,**extra):
        save(dict(state=state,phase=phase,pid=os.getpid(),timestamp=time.time(),**launch,
                  signature_sha256=sha(root/'navigation_pipeline_signature.json'),objective_achieved=False,**extra),root/'navigation_pipeline_status.json')
    while True:
        unchanged()
        try:service=service_status()
        except ObservationUnavailable as error:
            status('predecessor_observation_retry','confirmation',reason=str(error));time.sleep(poll_seconds);continue
        path=root/'post_pipeline_status.json';record=json.loads(path.read_text()) if path.exists() else None
        outcome=predecessor_outcome(service,record,identity)
        if outcome is not None:break
        status('waiting_for_complete_confirmation','confirmation',predecessor_pid=int(service['MainPID']));time.sleep(poll_seconds)
    if outcome==ADMITTED:load_candidate(root)
    def run_phase(name):
        module,args,needs_gpu=PHASES[name]
        while True:
            unchanged();path=completion_path(name,root)
            if needs_gpu and not(path is not None and path.exists()):
                while True:
                    pids=gpu_pids()
                    if not pids:break
                    status('waiting_for_GPU_idle',name,external_gpu_pids=pids);time.sleep(poll_seconds);unchanged()
            command=[sys.executable,'-u','-m','experiments.a2_waldo_landmark_belief.'+module,*args]
            environment=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            if not needs_gpu:environment['CUDA_VISIBLE_DEVICES']=''
            log=logs/(name+'.log');offset=log.stat().st_size if log.exists() else 0;status('running',name,log=str(log))
            with log.open('a') as stream:result=subprocess.run(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
            unchanged()
            if result.returncode:
                if needs_gpu and is_busy_rejection(log,offset):status('GPU_became_busy',name);continue
                raise RuntimeError(f'V4 navigation phase failed: {name}, exit={result.returncode}')
            result=validate_phase(name,root);status('completed_phase',name);return result
    final=execute_sequence(outcome,run_phase)
    status(final,'finished',confirmation_outcome=outcome,full_benchmark_completed=final=='full_benchmarks_and_report_complete')


def verify_terminal(root=None):
    root=Path(root) if root is not None else BASE/'revision_v4'
    require_cpu_gates(root)
    if json.loads((root/'navigation_pipeline_signature.json').read_text())!=cpu_signature(root):
        raise ValueError('Terminal navigation source snapshot belongs to another queue')
    service=service_status(NAV_UNIT);record=json.loads((root/'navigation_pipeline_status.json').read_text())
    if (service['LoadState']!='loaded' or int(service['MainPID'])!=0 or service['Result']!='success'
        or int(service['ExecMainStatus'])!=0 or (service['ActiveState'],service['SubState']) not in (('active','exited'),('inactive','dead'))
        or int(service['ExecMainPID'])!=record['native_main_pid'] or service['InvocationID']!=record['systemd_invocation_id']
        or record['execution_script_sha256']!=sha(ROOT/WRAPPER) or record['phase']!='finished'
        or record['signature_sha256']!=sha(root/'navigation_pipeline_signature.json')):
        raise ValueError('Actual V4 navigation service not successfully completed with matching identity')
    finished=record.get('state')=='full_benchmarks_and_report_complete'
    if record.get('state') not in ('full_benchmarks_and_report_complete','internal_confirmation_not_admitted','real_pilot_rejected_no_benchmark') or record.get('full_benchmark_completed') is not finished:
        raise ValueError('Unknown or inconsistent navigation terminal outcome')
    if finished:
        from .reference_v4_navigation_freeze import validate_frozen
        validate_frozen(root);require_report(ROOT/'reports/a2_execution_20261027/revision_v4_final',root)
    return record


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify-complete',action='store_true');args=parser.parse_args()
    root=BASE/'revision_v4'
    if args.verify_complete:print(json.dumps(verify_terminal(root)),flush=True);return
    with (root/'navigation_pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run_pipeline(root=root)
        except Exception as error:
            path=root/'navigation_pipeline_status.json';previous=json.loads(path.read_text()) if path.exists() else {}
            if previous.get('pid')!=os.getpid():previous={}
            save(dict(previous,state='failed',pid=os.getpid(),timestamp=time.time(),reason=str(error),objective_achieved=False,full_benchmark_completed=False),path);raise


if __name__=='__main__':main()
