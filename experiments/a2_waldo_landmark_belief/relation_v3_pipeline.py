"""Persistent serial scheduler: wait for real GPU idle; never stop other jobs."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
from .paths import ROOT,sha,save
from .collected_snapshot import BASE
from .relation_v3_train import code_hashes

PHASES=(
    ('cpu','relation_v3_preparation',('--cpu',),False),
    ('cuda','relation_v3_preparation',('--cuda',),True),
    ('a0','relation_v3_preflight',('--arm','a0'),True),
    ('off','relation_v3_preflight',('--arm','off'),True),
    ('on','relation_v3_preflight',('--arm','on'),True),
    ('runtime','relation_v3_preflight',('--arm','verify'),False),
    ('train','relation_v3_train',('--resume',),True),
    ('verify','relation_v3_verify',(),True),
    ('curves','relation_v3_verify',('--curves',),False))
MARKERS=dict(cpu=('cpu_gates.json','v3_cpu_tests_and_initial_state_verified'),
    cuda=('cuda_smoke.json','v3_real_TRAIN_CUDA_backward_passed'),
    runtime=('runtime_preflight.json','v3_real_TRAIN_runtime_preflight_passed'),
    train=('training/training_complete.json','v3_full20plus5_complete_artifact_verification_pending'),
    verify=('training/artifact_verification.json','v3_all25_artifacts_and_dev_verified'))


def phase_done(name,root,hashes,roles):
    root=Path(root)
    if name in ('a0','off','on'):
        path=root/'preflight'/f'{name}.json'
        if not path.exists():return False
        data=json.loads(path.read_text());signature=root/'preflight/signature.json';source=json.loads(signature.read_text())
        if data['signature_sha256']!=sha(signature) or data['episodes_completed']!=32 or source['code_sha256']!=hashes or source['roles_sha256']!=roles:
            raise ValueError('Existing functional arm changed: '+name)
        return True
    if name=='curves':
        path=root/'training/curve_manifest.json'
        if not path.exists():return False
        data=json.loads(path.read_text())
        if any(data[k]!=sha(root/'training'/v) for k,v in (('history_sha256','history.json'),('png_sha256','training_curves.png'),('svg_sha256','training_curves.svg'))):
            raise ValueError('Existing curves changed')
        return True
    relative,status=MARKERS[name];path=root/relative
    if not path.exists():return False
    data=json.loads(path.read_text())
    if data['status']!=status or data['code_sha256']!=hashes or data['roles_sha256']!=roles:raise ValueError('Existing proof changed: '+name)
    if name in ('train','verify'):
        if data['checkpoint_sha256']!=sha(root/'training/best.pt') or data['history_sha256']!=sha(root/'training/history.json'):
            raise ValueError('Existing trained artifact changed: '+name)
    if name=='verify':
        from .relation_v3_verify import validate_verified_proof
        validate_verified_proof(root/'training',hashes,roles)
    return True


def gpu_pids():
    from .gpu_execution_policy import blocking_gpu_pids
    return blocking_gpu_pids()


def is_busy_rejection(log,offset):
    with Path(log).open() as stream:
        stream.seek(offset);lines=stream.read().strip().splitlines()
    return bool(lines and lines[-1]=='ValueError: GPU occupied; no parallel A2 GPU work allowed')


def run_pipeline():
    root=BASE/'revision_v3';root.mkdir(parents=True,exist_ok=True);hashes=code_hashes();roles=sha(root/'roles.json')
    status_path=root/'pipeline_status.json';logs=root/'pipeline_logs';logs.mkdir(exist_ok=True)
    def status(state,phase,**extra):
        value=dict(state=state,phase=phase,pid=os.getpid(),timestamp=time.time(),code_sha256=hashes,roles_sha256=roles,**extra)
        save(value,status_path);print(json.dumps(value),flush=True)
    for name,module,args,needs_gpu in PHASES:
        while True:
            if code_hashes()!=hashes or sha(root/'roles.json')!=roles:raise ValueError('Pipeline code/roles changed; stop before next phase')
            if phase_done(name,root,hashes,roles):status('retained_completed_phase',name);break
            if needs_gpu:
                while True:
                    pids=gpu_pids()
                    if not pids:break
                    status('waiting_for_GPU_idle',name,external_gpu_pids=pids)
                    time.sleep(60)
                    if code_hashes()!=hashes or sha(root/'roles.json')!=roles:raise ValueError('Queued pipeline source changed')
            command=[sys.executable,'-u','-m','experiments.a2_waldo_landmark_belief.'+module,*args]
            environment=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            if not needs_gpu:environment['CUDA_VISIBLE_DEVICES']=''
            log=logs/(name+'.log');status('running',name,command=command,log=str(log))
            attempt_offset=log.stat().st_size if log.exists() else 0
            with log.open('a') as stream:
                result=subprocess.run(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
            if result.returncode:
                # A new unrelated job can start during CPU preparation. Retry
                # only the explicit idle guard rejection, never a generic error.
                if is_busy_rejection(log,attempt_offset):
                    status('GPU_became_busy_before_phase_finished',name,returncode=result.returncode);continue
                status('failed',name,returncode=result.returncode);raise RuntimeError('V3 phase failed: '+name)
            if not phase_done(name,root,hashes,roles):raise ValueError('Child exited without required complete proof: '+name)
            status('completed_phase',name);break
    marker=json.loads((root/'training/training_complete.json').read_text())
    status('training_verified_confirmation_pending' if marker['checkpoint_eligible'] else 'training_verified_no_eligible_DEV_gate',
           'finished',checkpoint_eligible=marker['checkpoint_eligible'],full_benchmark_completed=False)


def main():
    directory=BASE/'revision_v3';directory.mkdir(parents=True,exist_ok=True)
    with (directory/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run_pipeline()
        except Exception as error:
            path=directory/'pipeline_status.json'
            old=json.loads(path.read_text()) if path.exists() else {}
            if old.get('pid')!=os.getpid():old={}
            save(dict(old,state='failed',pid=os.getpid(),timestamp=time.time(),reason=str(error)),path)
            raise


if __name__=='__main__':main()
