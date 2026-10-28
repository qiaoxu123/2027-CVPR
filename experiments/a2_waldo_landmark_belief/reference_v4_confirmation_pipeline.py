"""Durable serial V4 handoff: complete training -> collection -> FIT -> VERIFY."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .paths import ROOT,sha,save
from .reference_overlay_context import BASE
from .reference_v4_confirmation_protocol import immutable
from .reference_v4_confirmation_preparation import cpu_signature,require_cpu_gates
from .reference_v4_calibration_data import FreshConfirmation,load_frozen_choice
from .reference_v4_train import training_signature
from .reference_v4_verify import validate_verified_proof
from .relation_v3_confirmation_pipeline import after_fit,predecessor_ready
from .relation_v3_pipeline import gpu_pids,is_busy_rejection


PREDECESSOR='graphnav-a2-reference-v4-pipeline-20261029.service'
SERVICE_FIELDS=('LoadState','ActiveState','SubState','MainPID','ExecMainStatus','Result')


class ObservationUnavailable(RuntimeError):
    """A failed observation cannot establish that the actual job stopped."""


def read_predecessor():
    command=['systemctl','--user','show',PREDECESSOR]
    for field in SERVICE_FIELDS:command.extend(('-p',field))
    try:
        result=subprocess.run(command,env=dict(os.environ,XDG_RUNTIME_DIR=f'/run/user/{os.getuid()}'),
                              text=True,capture_output=True,timeout=20)
    except (OSError,subprocess.TimeoutExpired) as error:
        raise ObservationUnavailable(str(error)) from error
    fields=dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
    if fields.get('LoadState')=='not-found':
        return fields  # predecessor_ready rejects authoritative missing units.
    if result.returncode or any(key not in fields for key in SERVICE_FIELDS):
        raise ObservationUnavailable('Predecessor state could not be read: '+result.stderr.strip())
    return fields


def admitted_training(root):
    root=Path(root)
    proof=validate_verified_proof(root/'training',training_signature())
    curves=json.loads((root/'training/curve_manifest.json').read_text())
    if any(curves.get(key)!=sha(root/'training'/name) for key,name in (
            ('history_sha256','history.json'),('png_sha256','training_curves.png'),('svg_sha256','training_curves.svg'))):
        raise ValueError('Verified V4 training curves changed')
    return proof


def validated_result(root):
    from .reference_v4_calibrate import analysis_signature,validate_existing
    source=FreshConfirmation(root)
    return validate_existing(source,analysis_signature(source))


def validate_completed_phase(root,name):
    if name=='collect':
        FreshConfirmation(root)
    elif name=='replay':
        from .reference_v4_sealed_replay import replay_manifest
        replay_manifest(FreshConfirmation(root))
    elif name=='fit':
        source=FreshConfirmation(root)
        decision=json.loads((root/'calibration/fit_decision.json').read_text())
        if decision['selected_parameters'] is None:
            validated_result(root)
        else:
            load_frozen_choice(root,source.checkpoint_sha256,source.roles_sha256,binding=source.state)
    elif name=='verify':
        validated_result(root)
    else:
        raise ValueError('Unknown V4 confirmation phase')


def run_pipeline(*,root=None,poll_seconds=300):
    root=Path(root) if root is not None else BASE/'revision_v4'
    if poll_seconds<=0:raise ValueError('Positive polling interval required')
    require_cpu_gates(root);signature=cpu_signature(root)
    immutable(signature,root/'post_pipeline_signature.json')
    logs=root/'post_pipeline_logs';logs.mkdir(exist_ok=True)
    def status(state,phase,**extra):
        value=dict(state=state,phase=phase,pid=os.getpid(),timestamp=time.time(),predecessor_unit=PREDECESSOR,
                   signature_sha256=sha(root/'post_pipeline_signature.json'),objective_achieved=False,
                   full_benchmark_completed=False,**extra)
        save(value,root/'post_pipeline_status.json');print(json.dumps(value),flush=True)
    def unchanged():
        if cpu_signature(root)!=signature:
            raise ValueError('Queued V4 post-training code/data/tests changed')
    while True:
        unchanged()
        try:
            service=read_predecessor()
        except ObservationUnavailable as error:
            status('predecessor_observation_retry','training',reason=str(error));time.sleep(poll_seconds);continue
        files_ready=(root/'training/artifact_verification.json').exists() and (root/'training/curve_manifest.json').exists()
        if predecessor_ready(service,files_ready):break
        status('waiting_for_full_training_and_verification','training',training_pid=int(service['MainPID']))
        time.sleep(poll_seconds)
    proof=admitted_training(root)
    if not proof['checkpoint_eligible']:
        status('full_training_no_eligible_DEV_gate_confirmation_unconsumed','finished');return
    commands=dict(collect=('reference_v4_confirmation',(),True),fit=('reference_v4_calibrate',('--stage','fit'),False),
                  replay=('reference_v4_sealed_replay',(),True),verify=('reference_v4_calibrate',('--stage','verify'),False))
    def run_phase(name):
        module,args,needs_gpu=commands[name]
        while True:
            unchanged();require_cpu_gates(root)
            marker=dict(collect=root/'confirmation/manifest.json',replay=root/'sealed_replay/manifest.json').get(name)
            if marker is not None and marker.exists():
                validate_completed_phase(root,name);status('retained_complete_phase',name);return
            if needs_gpu:
                while True:
                    pids=gpu_pids()
                    if not pids:break
                    status('waiting_for_GPU_idle',name,external_gpu_pids=pids);time.sleep(poll_seconds);unchanged()
            command=[sys.executable,'-u','-m','experiments.a2_waldo_landmark_belief.'+module,*args]
            environment=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            if not needs_gpu:environment['CUDA_VISIBLE_DEVICES']=''
            log=logs/(name+'.log');offset=log.stat().st_size if log.exists() else 0
            status('running',name,log=str(log))
            with log.open('a') as stream:
                result=subprocess.run(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
            if result.returncode:
                if is_busy_rejection(log,offset):status('GPU_became_busy',name);continue
                raise RuntimeError(f'V4 post-training phase failed: {name}, exit={result.returncode}')
            validate_completed_phase(root,name);status('completed_phase',name);return
    run_phase('collect');run_phase('fit')
    final=after_fit(root,run_phase,lambda:validated_result(root))
    status(final,'finished')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--poll-seconds',type=int,default=300);args=parser.parse_args()
    root=BASE/'revision_v4'
    with (root/'post_pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            run_pipeline(root=root,poll_seconds=args.poll_seconds)
        except Exception as error:
            path=root/'post_pipeline_status.json';previous=json.loads(path.read_text()) if path.exists() else {}
            if previous.get('pid')!=os.getpid():previous={}
            save(dict(previous,state='failed',pid=os.getpid(),timestamp=time.time(),reason=str(error),
                      objective_achieved=False,full_benchmark_completed=False),path)
            raise


if __name__=='__main__':
    main()
