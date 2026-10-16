"""Fixed original V1 checkpoint / 384 grid on every released split, resumable."""
import json
from pathlib import Path
import subprocess
import sys
import time
from graphnav.workspace_paths import PROJECT,current_paths
from graphnav.checkpointing import atomic_json,file_sha256


def main():
    root=Path('full_resolution_384');root.mkdir(exist_ok=True)
    base=current_paths(json.loads((PROJECT/'GraphNav/runs/graphnav_v1_stage1/full_seen/metrics.json').read_text())['args'])
    freeze=root/'frozen_resolution.json'
    if not freeze.exists():
        frozen=json.loads((PROJECT/'GraphNav/runs/graphnav_v1_stage1/frozen_final_candidate.json').read_text())
        frozen.update(grid_size=384,resolution_intervention=True,checkpoint=base['checkpoint'],frozen_before_any_full_384_score=time.time(),selection='User-requested fixed resolution; no weight/threshold selection by Unseen or Test')
        atomic_json(frozen,freeze)
    jobs=[]
    for split in ('val_seen','val_unseen','test_unseen'):
        output=root/split
        command=[sys.executable,'-m','scripts.evaluate_closedloop','--model','graphnav-v1',
            '--stage1-resolution-eval','--stage1-grid-size','384','--checkpoint',base['checkpoint'],
            '--sbf-source',base['sbf_source'],'--config',base['config'],'--data-root',base['data_root'],
            '--split',split,'--output',str(output),'--frozen-final-candidate',str(freeze)]
        if (output/'metrics.json').exists():continue
        if output.exists():command.append('--resume')
        job=dict(split=split,command=command,status='running',started=time.time());jobs.append(job)
        atomic_json(jobs,root/'queue.json');print(json.dumps(job),flush=True)
        with (root/(split+'.log')).open('a') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        job.update(returncode=result.returncode,finished=time.time(),status='complete' if result.returncode==0 else 'failed')
        atomic_json(jobs,root/'queue.json')
        if result.returncode:raise RuntimeError(f'{split} failed; preserved logs and episode traces')
        report=json.loads((output/'metrics.json').read_text())
        print(json.dumps(dict(split=split,metrics=report['metrics'])),flush=True)
    rows=[]
    for split in ('val_seen','val_unseen','test_unseen'):
        r=json.loads((root/split/'metrics.json').read_text());rows.append(dict(split=split,**r['metrics']))
    atomic_json(dict(grid_size=384,weights_sha256=file_sha256(base['checkpoint']),results=rows),root/'all_splits.json')


if __name__=='__main__':main()
