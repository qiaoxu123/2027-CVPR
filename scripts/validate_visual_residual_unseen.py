"""Run only missing frozen residual Unseen controls, then complete paired report."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.workspace_paths import PROJECT,current_paths

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'visual_residual_unseen_validation'
CONTROLS=('real','no-belief','zero','shuffle')
BASE_SHA='00f2acc0e3d0cd2c141dbf852c1205fedd218a174ea1d5528edba67217d77c5a'
RESIDUAL_SHA='5fc1a3e115b71d14fe1e8fbaf3ed29c1863d8b2bc080455d89f7a4e503d25f11'


def completed_valid(directory,split,control):
    p=directory/'metrics.json'
    if not p.exists():return False
    report=json.loads(p.read_text())
    assert report['checkpoint_sha256']==BASE_SHA and report['calibrated_residual_sha256']==RESIDUAL_SHA
    assert report['split']==split and report['args']['stage1_visual_control']==control
    assert report['stage1_inference_controls']['grid_size']==28
    assert report['stage1_inference_controls']['stop_threshold']==.3
    assert report['metrics']['episodes']==(2470 if split=='val_seen' else 2697)
    assert sum(1 for _ in (directory/'predictions.jsonl').open())==report['metrics']['episodes']
    return True


def main():
    OUT.mkdir(exist_ok=True)
    lock=(OUT/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    official=PROJECT/'GraphNav/runs/graphnav_v1_stage1'
    base_report=json.loads((official/'full_unseen/metrics.json').read_text())
    base=current_paths(base_report['args']);head=ROOT/'calibrated_residual/best.pt'
    assert file_sha256(base['checkpoint'])==BASE_SHA and file_sha256(head)==RESIDUAL_SHA
    seen={}
    for control in CONTROLS:
        directory=ROOT/'visual_evaluations'/f'calibrated_residual_full_seen_{control}'
        assert completed_valid(directory,'val_seen',control)
        seen[control]=str(directory)
    freeze=json.loads((official/'frozen_final_candidate.json').read_text())
    freeze.update(validation_kind='calibrated-residual-val-unseen-v1',checkpoint=base['checkpoint'],
                  calibrated_residual_checkpoint=str(head),calibrated_residual_sha256=RESIDUAL_SHA,
                  allowed_visual_controls=list(CONTROLS),grid_size=28,stop_threshold=.3,
                  expected_episode_manifest_sha256=file_sha256(official/'full_unseen/episodes.json'),
                  input_sha256=base_report['input_sha256'],bootstrap_replicates=20000,bootstrap_seed=20261017,
                  training=False,selection='Existing Calibrated checkpoint selected only by Train Dev; fixed four controls, no Unseen/Test selection',
                  seen_reused=seen)
    frozen=OUT/'frozen_validation.json'
    if frozen.exists():assert json.loads(frozen.read_text())==freeze
    else:atomic_json(freeze,frozen)
    snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=True)
    hashes={}
    for p in [ROOT/'scripts/evaluate_closedloop.py',ROOT/'scripts/validate_visual_residual_unseen.py',
              ROOT/'scripts/analyze_visual_residual_unseen.py',*sorted((ROOT/'graphnav').glob('*.py'))]:
        if not p.exists():continue
        rel=p.relative_to(ROOT);hashes[str(rel)]=file_sha256(p)
        target=snapshot/rel;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert target.read_bytes()==p.read_bytes(),'Do not mutate launched evaluation code'
        else:target.write_bytes(p.read_bytes())
    atomic_json(hashes,OUT/'source_sha256.json')
    atomic_json(dict(time=time.time(),branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
                     commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                     device=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv'],text=True),
                     bootstrap_replicates=20000,bootstrap_seed=20261017,seen_reused=seen),OUT/'launch_record.json')
    environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                     HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
    queue=[];processes={};logs={}
    for control in CONTROLS:
        directory=ROOT/'visual_evaluations'/f'calibrated_residual_full_unseen_{control}'
        command=[sys.executable,'-u','-m','scripts.evaluate_closedloop','--model','graphnav-v1',
                 '--checkpoint',base['checkpoint'],'--sbf-source',base['sbf_source'],'--config',base['config'],
                 '--data-root',base['data_root'],'--split','val_unseen','--stage1-residual-checkpoint',str(head),
                 '--stage1-visual-control',control,'--frozen-final-candidate',str(frozen),'--output',str(directory)]
        job=dict(control=control,output=str(directory),command=command,started=time.time())
        queue.append(job)
        if completed_valid(directory,'val_unseen',control):job.update(status='reused',returncode=0);continue
        if directory.exists():command.append('--resume')
        log=(OUT/f'{control}.log').open('a');logs[control]=log
        process=subprocess.Popen(command,env=environment,stdout=log,stderr=subprocess.STDOUT)
        processes[control]=process;job.update(status='running',pid=process.pid)
    atomic_json(queue,OUT/'queue.json')
    while processes:
        progress={}
        for job in queue:
            trace=Path(job['output'])/'predictions.jsonl'
            progress[job['control']]=dict(completed=sum(1 for _ in trace.open()) if trace.exists() else 0,total=2697,status=job['status'])
        atomic_json(dict(updated=time.time(),controls=progress),OUT/'progress.json')
        for control,process in list(processes.items()):
            rc=process.poll()
            if rc is None:continue
            job=next(j for j in queue if j['control']==control)
            job.update(returncode=rc,status='complete' if rc==0 else 'failed',finished=time.time())
            logs[control].close();del processes[control];atomic_json(queue,OUT/'queue.json')
        if processes:time.sleep(10)
    if any(j['status']=='failed' for j in queue):raise RuntimeError('Evaluation failed; preserve outputs and repair before resuming')
    for control in CONTROLS:
        assert completed_valid(ROOT/'visual_evaluations'/f'calibrated_residual_full_unseen_{control}','val_unseen',control)
    assert file_sha256(head)==RESIDUAL_SHA and file_sha256(base['checkpoint'])==BASE_SHA
    subprocess.run([sys.executable,'-u','-m','scripts.analyze_visual_residual_unseen'],env=environment,check=True)
    atomic_json(dict(complete=True,finished=time.time(),controls=4,episodes_per_control=2697),OUT/'complete.json')
    atomic_json(dict(updated=time.time(),complete=True,controls={c:dict(completed=2697,total=2697,status='complete') for c in CONTROLS}),OUT/'progress.json')


if __name__=='__main__':
    try:main()
    except Exception as error:
        OUT.mkdir(exist_ok=True);atomic_json(dict(error=str(error),time=time.time()),OUT/'failure.json');raise
