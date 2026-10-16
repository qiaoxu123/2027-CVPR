"""Frozen-weight Seen-only arrival-guard sweep; probability threshold is fixed."""
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.workspace_paths import PROJECT,current_paths

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'stop_guard_results'
VARIANTS=[('1mm',.001),('1m',1.),('5m',5.),('10m',10.),('20m',20.),('disabled',-1.)]


def summarize():
    rows=[];paired={}
    for scope,count in [('seen128',128),('full_seen',2470)]:
        baseline_path=OUT/f'{scope}_1mm'
        if not (baseline_path/'metrics.json').exists():continue
        baseline=json.loads((baseline_path/'metrics.json').read_text())
        baseline_rows=None
        for name,radius in VARIANTS:
            directory=OUT/f'{scope}_{name}';p=directory/'metrics.json'
            if not p.exists():continue
            d=json.loads(p.read_text());m=d['metrics']
            assert m['episodes']==count
            assert d['checkpoint_sha256']==baseline['checkpoint_sha256']
            assert d['episode_manifest_sha256']==baseline['episode_manifest_sha256']
            rows.append(dict(scope=scope,arrival_guard_m=radius,episodes=count,stop_probability_threshold=.3,
                             SR_percent=m['SR']*100,SPL_percent=m['SPL']*100,NE_m=m['NE'],steps=m['steps'],
                             false_stop_per_stop_percent=m['False_Stop_Rate']*100 if m['False_Stop_Rate'] is not None else None,
                             false_stop_episode_percent=m['false_stop_episode_fraction']*100,policy_stop_fraction=m['policy_stop']))
            if name=='1mm':continue
            if baseline_rows is None:baseline_rows=[json.loads(line) for line in (baseline_path/'predictions.jsonl').open()]
            predictions=[json.loads(line) for line in (directory/'predictions.jsonl').open()]
            assert [r['index'] for r in predictions]==[r['index'] for r in baseline_rows]
            diff=np.asarray([b['metrics']['SR']-a['metrics']['SR'] for a,b in zip(baseline_rows,predictions)])
            rng=np.random.default_rng(0)
            samples=np.concatenate([diff[rng.integers(len(diff),size=(250,len(diff)))].mean(1) for _ in range(40)])
            paired[f'{scope}_{name}']=dict(episodes=count,sr_gain_pp=float(diff.mean()*100),
                                          sr_paired_bootstrap_95ci_pp=(np.quantile(samples,[.025,.975])*100).tolist(),
                                          helped=int((diff>0).sum()),hurt=int((diff<0).sum()))
    if rows:
        with (OUT/'comparison.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    atomic_json(paired,OUT/'paired_comparison.json')


def main():
    OUT.mkdir(exist_ok=True);lock=(OUT/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    base=current_paths(json.loads((PROJECT/'GraphNav/runs/graphnav_v1_stage1/full_seen/metrics.json').read_text())['args'])
    sha=file_sha256(base['checkpoint'])
    assert sha=='00f2acc0e3d0cd2c141dbf852c1205fedd218a174ea1d5528edba67217d77c5a'
    plan=dict(variants=VARIANTS,grid_size=28,stop_probability_threshold=.3,altitude_guard='unchanged ground+10..100m',
              training=False,checkpoint_sha256=sha,context='original instruction references only',
              scopes=['Val-Seen evenly spaced 128','complete Val-Seen 2470'],
              selection='No automatic model/controller replacement; no Unseen/Test tuning',
              source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    path=OUT/'frozen_plan.json'
    if path.exists():assert json.loads(path.read_text())==json.loads(json.dumps(plan))
    else:atomic_json(plan,path)
    environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                     HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
    queue=[]
    for scope,count in [('seen128',128),('full_seen',2470)]:
        for name,radius in VARIANTS:
            destination=OUT/f'{scope}_{name}'
            command=[sys.executable,'-u','-m','scripts.evaluate_closedloop','--model','graphnav-v1',
                     '--stage1-diagnostic','--stage1-arrival-guard-m',str(radius),'--stop-threshold','.3',
                     '--checkpoint',base['checkpoint'],'--sbf-source',base['sbf_source'],'--config',base['config'],
                     '--data-root',base['data_root'],'--split','val_seen','--output',str(destination)]
            if scope=='seen128':command+=['--max-episodes','128']
            job=dict(name=f'{scope}_{name}',arrival_guard_m=radius,total=count,command=command,started=time.time())
            queue.append(job)
            if (destination/'metrics.json').exists():job.update(status='complete',returncode=0)
            else:
                if destination.exists():command.append('--resume')
                with (OUT/f'{scope}_{name}.log').open('a') as log:
                    child=subprocess.Popen(command,env=environment,stdout=log,stderr=subprocess.STDOUT)
                    job.update(pid=child.pid,status='running');atomic_json(queue,OUT/'queue.json')
                    while child.poll() is None:
                        trace=destination/'predictions.jsonl'
                        n=sum(1 for _ in trace.open()) if trace.exists() else 0
                        atomic_json(dict(job=job['name'],pid=child.pid,completed=n,total=count,updated=time.time()),OUT/'progress.json')
                        time.sleep(10)
                    job.update(returncode=child.returncode,status='complete' if child.returncode==0 else 'failed')
                    atomic_json(queue,OUT/'queue.json')
                    if child.returncode:raise RuntimeError(f'{job["name"]} failed; preserve output and inspect logs')
            assert file_sha256(base['checkpoint'])==sha
            job['finished']=time.time();atomic_json(queue,OUT/'queue.json');summarize();print(json.dumps(job),flush=True)
    atomic_json(dict(complete=True,finished=time.time(),variants=6,episodes_per_variant=2470),OUT/'complete.json')
    atomic_json(dict(complete=True,updated=time.time()),OUT/'progress.json')


if __name__=='__main__':
    try:main()
    except Exception as error:
        OUT.mkdir(exist_ok=True);atomic_json(dict(error=str(error),time=time.time()),OUT/'failure.json');raise
