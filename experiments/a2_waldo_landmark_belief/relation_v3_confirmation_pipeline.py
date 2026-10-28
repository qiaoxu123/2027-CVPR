"""Persistent post-training serial collection/FIT/replay/VERIFY scheduler."""
import argparse,fcntl,hashlib,json,os,re,shutil,subprocess,sys,time
from pathlib import Path
from .paths import ROOT,sha,save
from .collected_snapshot import BASE
from .relation_v3_calibration_data import calibration_code_hashes,FreshConfirmation
from .relation_v3_gate_fit import validate_parameters
from .relation_v3_train import code_hashes as training_code_hashes
from .relation_v3_preparation import test_hashes as training_test_hashes
from .relation_v3_verify import validate_verified_proof
from .relation_v3_pipeline import gpu_pids,is_busy_rejection


def post_test_hashes():return {str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'tests_confirmation_v3').rglob('*.py'))}


def cpu_signature(root):
    return dict(implementation=calibration_code_hashes(),post_test_source_sha256=post_test_hashes(),
        training_code_sha256=training_code_hashes(),training_test_source_sha256=training_test_hashes(),
        roles_sha256=sha(root/'roles.json'),population_sha256=sha(root/'confirmation_population.json'),protocol_sha256=sha(root/'confirmation_protocol.json'))


def require_cpu_gates(root=BASE/'revision_v3'):
    root=Path(root);proof=json.loads((root/'posttraining_cpu_gates.json').read_text());expected=cpu_signature(root)
    if (proof['status']!='v3_posttraining_CPU_gates_passed' or any(proof[k]!=v for k,v in expected.items())
        or proof['test_log_sha256']!=sha(root/'posttraining_cpu/cpu_tests.log')):raise ValueError('Current post-training CPU gate required')
    return proof


def cpu_gates():
    root=BASE/'revision_v3';signature=cpu_signature(root);path=root/'posttraining_cpu_gates.json'
    if path.exists():
        try:require_cpu_gates(root);print('Retained matching post-training CPU gate',flush=True);return
        except ValueError:
            if (root/'calibration/fit_decision.json').exists():raise ValueError('Cannot replace CPU proof after FIT choice is frozen')
            archive=root/'posttraining_cpu/archive';archive.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,archive/(sha(path)+'.json'))
    directory=root/'posttraining_cpu';directory.mkdir(exist_ok=True);log=directory/'cpu_tests.log'
    if log.exists():
        archive=directory/'archive';archive.mkdir(exist_ok=True);shutil.copyfile(log,archive/(sha(log)+'.log'))
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    with log.open('w') as stream:
        result=subprocess.run([sys.executable,'-m','pytest','-q','tests','tests_confirmation_v3'],cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
    match=re.search(r'\d+ passed[^\n]*',log.read_text())
    if result.returncode or match is None or signature!=cpu_signature(root):raise ValueError('Post-training complete CPU test gate failed or source changed')
    proof=dict(signature,status='v3_posttraining_CPU_gates_passed',test_log_sha256=sha(log),unit_test_summary=match.group(),no_CUDA_used=True,no_new_confirmation_data_consumed=True)
    save(proof,path)
    save(dict(status=proof['status'],unit_test_summary=proof['unit_test_summary'],gate_sha256=sha(path),implementation=signature['implementation'],
        roles_sha256=signature['roles_sha256'],population_sha256=signature['population_sha256'],no_CUDA_used=True),ROOT/'reports/a2_execution_20261027/RELATION_V3_POSTTRAINING_CPU_GATES.json')
    print(json.dumps(dict(status=proof['status'],unit_test_summary=proof['unit_test_summary'])),flush=True)


def after_fit(root,run_phase,validated_result):
    root=Path(root);decision=json.loads((root/'calibration/fit_decision.json').read_text())
    if decision['selected_parameters'] is None:
        result=validated_result()
        if result['enabled'] or result['verification'] is not None or result['verification_performance_evaluated']:raise ValueError('No-FIT outcome improperly consumed/enabled VERIFY')
        return 'FIT_no_candidate_VERIFY_unconsumed'
    validate_parameters(decision['selected_parameters'])
    if not (root/'calibration/candidate_calibration.json').exists():run_phase('replay');run_phase('verify')
    result=validated_result()
    if result['verification'] is None or result['verification']['groups']!=209:raise ValueError('Complete209 VERIFY required')
    return 'calibration_passed_real_pilot_pending' if result['enabled'] else 'VERIFY_rejected_no_forced_enable'


def predecessor_ready(service,files_ready):
    if service['LoadState']!='loaded' or service['ActiveState']=='failed' or int(service['ExecMainStatus'])!=0 or service['Result']!='success':
        raise ValueError('Training predecessor missing or failed')
    if int(service['MainPID'])>0 or service['SubState'] in ('running','start','start-pre','start-post','stop','stop-sigterm','stop-sigkill'):return False
    if int(service['MainPID'])==0 and ((service['ActiveState']=='active' and service['SubState']=='exited') or (service['ActiveState']=='inactive' and service['SubState']=='dead')):
        if not files_ready:raise ValueError('Training predecessor terminal without complete verified files')
        return True
    raise ValueError('Unknown training predecessor state')


def run_pipeline():
    root=BASE/'revision_v3';require_cpu_gates(root);signature=cpu_signature(root);logs=root/'post_pipeline_logs';logs.mkdir(exist_ok=True)
    status_path=root/'post_pipeline_status.json';unit='graphnav-a2-relation-v3-pipeline-20261028.service'
    def status(state,phase,**extra):
        value=dict(state=state,phase=phase,pid=os.getpid(),timestamp=time.time(),**extra)
        save(value,status_path);print(json.dumps(value),flush=True)
    def unchanged():
        if cpu_signature(root)!=signature:raise ValueError('Queued post-training source/roles/tests changed')
    while True:
        unchanged();environment=dict(os.environ,XDG_RUNTIME_DIR=f'/run/user/{os.getuid()}')
        text=subprocess.check_output(['systemctl','--user','show',unit,'-p','LoadState','-p','ActiveState','-p','SubState','-p','MainPID','-p','ExecMainStatus','-p','Result'],text=True,env=environment)
        service=dict(line.split('=',1) for line in text.strip().splitlines())
        files_ready=(root/'training/artifact_verification.json').exists() and (root/'training/curve_manifest.json').exists()
        if predecessor_ready(service,files_ready):break
        status('waiting_for_full_training_and_verification','training',training_pid=int(service['MainPID']));time.sleep(60)
    proof=validate_verified_proof(root/'training',training_code_hashes(),sha(root/'roles.json'))
    if not proof['checkpoint_eligible']:status('full_training_no_eligible_DEV_gate_confirmation_unconsumed','finished');return
    curve=json.loads((root/'training/curve_manifest.json').read_text())
    if any(curve[k]!=sha(root/'training'/v) for k,v in (('history_sha256','history.json'),('png_sha256','training_curves.png'),('svg_sha256','training_curves.svg'))):raise ValueError('Verified training curves changed')
    commands=dict(collect=('relation_v3_confirmation',(),True),fit=('relation_v3_calibrate',('--stage','fit'),False),
        replay=('relation_v3_sealed_replay',(),True),verify=('relation_v3_calibrate',('--stage','verify'),False))
    def run_phase(name):
        module,args,needs_gpu=commands[name]
        while True:
            unchanged();require_cpu_gates(root)
            if name=='collect' and (root/'confirmation/manifest.json').exists():FreshConfirmation();status('retained_complete_fresh163','collect');return
            if name=='replay' and (root/'sealed_replay/manifest.json').exists():
                from .relation_v3_sealed_replay import replay_manifest
                replay_manifest(FreshConfirmation());status('retained_complete_sealed130_replay','replay');return
            if needs_gpu:
                while True:
                    pids=gpu_pids()
                    if not pids:break
                    status('waiting_for_GPU_idle',name,external_gpu_pids=pids);time.sleep(60);unchanged()
            command=[sys.executable,'-u','-m','experiments.a2_waldo_landmark_belief.'+module,*args]
            environment=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            if not needs_gpu:environment['CUDA_VISIBLE_DEVICES']=''
            log=logs/(name+'.log');offset=log.stat().st_size if log.exists() else 0;status('running',name,log=str(log))
            with log.open('a') as stream:result=subprocess.run(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
            if result.returncode:
                if is_busy_rejection(log,offset):status('GPU_became_busy',name);continue
                raise RuntimeError(f'Post-training phase failed: {name}, exit={result.returncode}')
            status('completed_phase',name);return
    run_phase('collect');run_phase('fit')
    def validated_result():
        from .relation_v3_calibrate import analysis_signature,validate_existing
        source=FreshConfirmation();return validate_existing(source,analysis_signature(source))
    final=after_fit(root,run_phase,validated_result);status(final,'finished',full_benchmark_completed=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cpu-gates',action='store_true');args=parser.parse_args()
    if args.cpu_gates:cpu_gates();return
    root=BASE/'revision_v3'
    with (root/'post_pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run_pipeline()
        except Exception as error:
            path=root/'post_pipeline_status.json';old=json.loads(path.read_text()) if path.exists() else {}
            if old.get('pid')!=os.getpid():old={}
            save(dict(old,state='failed',pid=os.getpid(),timestamp=time.time(),reason=str(error)),path);raise


if __name__=='__main__':main()
