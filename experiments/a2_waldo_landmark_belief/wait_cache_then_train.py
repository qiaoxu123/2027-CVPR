"""CPU-only waiter; acquire the shared GPU lock only after cache success.

Never restart a failed/missing job. Never start a second GPU workload. The
updated-model real CUDA smoke and full20+5 training execute strictly in order.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime,timezone
from .paths import ROOT,sha,save


def cache_ready(properties):
    if properties['LoadState']!='loaded':
        raise RuntimeError('Cache service handle is missing; inspect artifacts before continuing')
    if properties['ActiveState'] in ('active','activating','deactivating','reloading'):
        return False
    if properties['ActiveState']!='inactive' or properties['ExecMainStatus']!='0' or properties['Result']!='success':
        raise RuntimeError('Cache service did not finish successfully; do not restart or launch training')
    return True


def verify_pin(pin):
    package=ROOT/'experiments/a2_waldo_landmark_belief'
    for name,digest in pin['modules_sha256'].items():
        if sha(package/name)!=digest:
            raise RuntimeError(f'Reviewed training implementation changed while queued: {name}')
    signature=ROOT/'runs/a2_waldo_landmark_20261027/cache/signature.json'
    if sha(signature)!=pin['cache_signature_sha256']:
        raise RuntimeError('Queued cache provenance changed')


def spawn_locked(command,lock_fd,**kwargs):
    """Child retains the same flock even if its CPU-only waiter dies."""
    return subprocess.Popen(command,pass_fds=(lock_fd,),**kwargs)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pin',type=Path,required=True)
    args=parser.parse_args()
    pin=json.loads(args.pin.read_text())
    output=ROOT/'runs/a2_waldo_landmark_20261027'
    state=dict(status='waiting_for_cache',cache_unit=pin['cache_unit'],pin_sha256=sha(args.pin),main_pid=os.getpid())
    def publish(**changes):
        state.update(changes,at_utc=datetime.now(timezone.utc).isoformat())
        save(state,output/'training_queue_state.json')
        print(json.dumps(state),flush=True)
    def run_child(module,*arguments,lock_fd):
        verify_pin(pin)
        command=[sys.executable,'-u','-m',f'experiments.a2_waldo_landmark_belief.{module}',*map(str,arguments)]
        child=spawn_locked(command,lock_fd,cwd=ROOT)
        try:
            publish(status=module,child_pid=child.pid)
            code=child.wait()
        except BaseException:
            child.terminate()
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill();child.wait()
            raise
        if code!=0:
            raise RuntimeError(f'{module} failed; inspect logs before any restart')
    try:
        verify_pin(pin)
        while True:
            env=dict(os.environ,XDG_RUNTIME_DIR=os.environ.get('XDG_RUNTIME_DIR',f'/run/user/{os.getuid()}'))
            result=subprocess.run(['systemctl','--user','show',pin['cache_unit'],
                                   '--property=LoadState,ActiveState,ExecMainStatus,Result,MainPID'],
                                  env=env,capture_output=True,text=True,check=True)
            properties=dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
            if cache_ready(properties):
                break
            publish(cache_properties=properties)
            time.sleep(30)
        with (output/'gpu.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],
                               capture_output=True,text=True,check=True)
            if gpu.stdout.strip():raise RuntimeError('GPU compute process still active; refusing parallel work')
            cache=output/'cache'
            if json.loads((cache/'manifest.json').read_text()).get('complete') is not True:
                raise RuntimeError('Full-cache completion manifest missing')
            run_child('verify_rows','--cache',cache,lock_fd=lock.fileno())
            run_child('train','--cache',cache,lock_fd=lock.fileno())
            complete=output/'training/training_complete.json'
            if not complete.exists():raise RuntimeError('Training child exited without completion artifacts')
            publish(status='full20plus5_training_complete_calibration_pending',
                    training_complete_sha256=sha(complete),child_pid=None)
    except Exception as error:
        publish(status='failed_requires_inspection',error=type(error).__name__+': '+str(error))
        raise


if __name__=='__main__':
    main()
