"""Fixed 28-grid, no-training Val-Seen context experiment and paired summaries."""
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from graphnav.checkpointing import atomic_json, file_sha256
from graphnav.workspace_paths import PROJECT, current_paths


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'landmark_context_results'
JOBS = [('smoke_references', 'references', 4),
        ('smoke_masked', 'all_named_masked', 4),
        ('smoke_all_named', 'all_named', 4),
        ('seen128_references', 'references', 128),
        ('seen128_all_named', 'all_named', 128),
        ('full_seen_references', 'references', None),
        ('full_seen_all_named', 'all_named', None)]


def summarize():
    rows=[]
    for name,mode,count in JOBS:
        p=OUT/name/'metrics.json'
        if not p.exists():continue
        d=json.loads(p.read_text());m=d['metrics']
        rows.append(dict(job=name,context=mode,episodes=m['episodes'],SR_percent=m['SR']*100,
                         SPL_percent=m['SPL']*100,NE_m=m['NE'],steps=m['steps'],
                         seconds=d['seconds'],checkpoint_sha256=d['checkpoint_sha256'],
                         episode_manifest_sha256=d['episode_manifest_sha256'],
                         initial_static_R1_percent=d['goal_retrieval']['initial_static_R@1/20m']*100))
    if rows:
        with (OUT/'comparison.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    paired={}
    for scope in ('seen128','full_seen'):
        paths=[OUT/f'{scope}_{mode}' for mode in ('references','all_named')]
        if not all((p/'metrics.json').exists() for p in paths):continue
        reports=[json.loads((p/'metrics.json').read_text()) for p in paths]
        assert reports[0]['checkpoint_sha256']==reports[1]['checkpoint_sha256']
        assert reports[0]['episode_manifest_sha256']==reports[1]['episode_manifest_sha256']
        predictions=[[json.loads(line) for line in (p/'predictions.jsonl').open()] for p in paths]
        a,b=predictions;assert [x['index'] for x in a]==[x['index'] for x in b]
        diff=np.asarray([y['metrics']['SR']-x['metrics']['SR'] for x,y in zip(a,b)])
        rng=np.random.default_rng(0)
        samples=np.concatenate([diff[rng.integers(len(diff),size=(250,len(diff)))].mean(1) for _ in range(40)])
        paired[scope]=dict(episodes=len(a),sr_gain_pp=float(diff.mean()*100),
                          sr_paired_bootstrap_95ci_pp=(np.quantile(samples,[.025,.975])*100).tolist(),
                          helped=int((diff>0).sum()),hurt=int((diff<0).sum()),
                          spl_gain_pp=(reports[1]['metrics']['SPL']-reports[0]['metrics']['SPL'])*100,
                          ne_change_m=reports[1]['metrics']['NE']-reports[0]['metrics']['NE'])
    atomic_json(paired,OUT/'paired_comparison.json')


def check_masked_smoke():
    paths=[OUT/name for name in ('smoke_references','smoke_masked')]
    reports=[json.loads((p/'metrics.json').read_text()) for p in paths]
    assert reports[0]['episode_manifest_sha256']==reports[1]['episode_manifest_sha256']
    rows=[[json.loads(line) for line in (p/'predictions.jsonl').open()] for p in paths]
    outcome_changes=[];max_final_difference=0.
    for a,b in zip(*rows):
        assert a['index']==b['index']
        if a['metrics']['SR']!=b['metrics']['SR'] or a['termination']!=b['termination']:
            outcome_changes.append(a['index'])
        max_final_difference=max(max_final_difference,float(np.max(np.abs(np.asarray(a['poses'][-1])-np.asarray(b['poses'][-1])))))
    audit=dict(episodes=len(rows[0]),outcome_changes=outcome_changes,max_final_pose_difference=max_final_difference)
    atomic_json(audit,OUT/'masked_smoke_check.json')
    if outcome_changes or max_final_difference>1.:
        raise ValueError('Masked context changed baseline materially; inspect smoke evidence before continuing')


def main():
    OUT.mkdir(exist_ok=True)
    lock=(OUT/'queue.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    base=current_paths(json.loads((PROJECT/'GraphNav/runs/graphnav_v1_stage1/full_seen/metrics.json').read_text())['args'])
    checkpoint_sha=file_sha256(base['checkpoint'])
    assert checkpoint_sha=='00f2acc0e3d0cd2c141dbf852c1205fedd218a174ea1d5528edba67217d77c5a'
    plan=dict(grid_size=28,stop_threshold=.3,checkpoint=base['checkpoint'],checkpoint_sha256=checkpoint_sha,
              training=False,selection='fixed input interventions; no model/threshold selection; Val-Seen only',
              intervention='Append all public named-map entities to candidate-entity attention. Keep reference priors/rules unchanged; do not create a raster background encoder or add untrained flag weights.',
              named_policy='SBF all_landmarks largest exact-name polygon union original reference fragments; no spatial truncation or GT access',
              jobs=JOBS,source_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip())
    p=OUT/'frozen_plan.json'
    if p.exists():assert json.loads(p.read_text())==json.loads(json.dumps(plan))
    else:atomic_json(plan,p)
    environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                     HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
    queue=[]
    for name,mode,count in JOBS:
        if name=='smoke_all_named':check_masked_smoke()
        destination=OUT/name
        command=[sys.executable,'-u','-m','scripts.evaluate_closedloop','--model','graphnav-v1',
                 '--stage1-diagnostic','--landmark-context',mode,'--stop-threshold','.3',
                 '--checkpoint',base['checkpoint'],'--sbf-source',base['sbf_source'],
                 '--config',base['config'],'--data-root',base['data_root'],'--split','val_seen','--output',str(destination)]
        if count is not None:command+=['--max-episodes',str(count)]
        job=dict(name=name,context=mode,total=count or 2470,command=command,started=time.time())
        queue.append(job)
        if (destination/'metrics.json').exists():job.update(status='complete',returncode=0)
        else:
            if destination.exists():command.append('--resume')
            with (OUT/f'{name}.log').open('a') as log:
                process=subprocess.Popen(command,env=environment,stdout=log,stderr=subprocess.STDOUT)
                job.update(pid=process.pid,status='running');atomic_json(queue,OUT/'queue.json')
                while process.poll() is None:
                    trace=destination/'predictions.jsonl'
                    completed=sum(1 for _ in trace.open()) if trace.exists() else 0
                    atomic_json(dict(job=name,pid=process.pid,completed=completed,total=count or 2470,updated=time.time()),OUT/'progress.json')
                    time.sleep(10)
                job.update(returncode=process.returncode,status='complete' if process.returncode==0 else 'failed')
            if process.returncode:
                atomic_json(queue,OUT/'queue.json')
                raise RuntimeError(f'{name} failed; inspect preserved log and resume')
        assert file_sha256(base['checkpoint'])==checkpoint_sha
        job['finished']=time.time();atomic_json(queue,OUT/'queue.json');summarize()
        print(json.dumps(job),flush=True)
    atomic_json(dict(complete=True,finished=time.time()),OUT/'complete.json')
    atomic_json(dict(complete=True,completed=2470,total=2470,updated=time.time()),OUT/'progress.json')


if __name__=='__main__':
    try:main()
    except Exception as error:
        OUT.mkdir(exist_ok=True)
        atomic_json(dict(error=str(error),time=time.time()),OUT/'failure.json')
        raise
