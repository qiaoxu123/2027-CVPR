"""Run each fixed 384 split independently; no tuning and no shared episode state."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from graphnav.workspace_paths import PROJECT,current_paths
from graphnav.checkpointing import atomic_json


def main():
    root=Path('full_resolution_384');root.mkdir(exist_ok=True)
    base=current_paths(json.loads((PROJECT/'GraphNav/runs/graphnav_v1_stage1/full_seen/metrics.json').read_text())['args'])
    freeze=root/'frozen_resolution.json'
    if not freeze.exists():raise ValueError('Resolution must be frozen before scoring')
    environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    jobs=[];processes={};logs={}
    for split in ('val_seen','val_unseen','test_unseen'):
        output=root/split
        if (output/'metrics.json').exists():continue
        command=[sys.executable,'-m','scripts.evaluate_closedloop','--model','graphnav-v1','--stage1-resolution-eval','--stage1-grid-size','384','--checkpoint',base['checkpoint'],'--sbf-source',base['sbf_source'],'--config',base['config'],'--data-root',base['data_root'],'--split',split,'--output',str(output),'--frozen-final-candidate',str(freeze)]
        if output.exists():command.append('--resume')
        log=(root/(split+'.log')).open('a');logs[split]=log
        process=subprocess.Popen(command,env=environment,stdout=log,stderr=subprocess.STDOUT)
        processes[split]=process;jobs.append(dict(split=split,pid=process.pid,command=command,status='running',started=time.time()))
    atomic_json(jobs,root/'parallel_queue.json')
    while processes:
        for split,process in list(processes.items()):
            result=process.poll()
            if result is None:continue
            job=next(j for j in jobs if j['split']==split);job.update(status='complete' if result==0 else 'failed',returncode=result,finished=time.time())
            logs[split].close();del processes[split];atomic_json(jobs,root/'parallel_queue.json')
        time.sleep(5)
    if any(j['status']=='failed' for j in jobs):raise RuntimeError('See preserved per-split logs; resumable')
    reports={split:json.loads((root/split/'metrics.json').read_text()) for split in ('val_seen','val_unseen','test_unseen')}
    atomic_json({split:report['metrics'] for split,report in reports.items()},root/'all_splits.json')


if __name__=='__main__':main()
