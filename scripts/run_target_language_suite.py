"""Fixed-budget target-language ablation; no reference-role or model changes."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import sys
from graphnav.checkpointing import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve previous suite')
    a.output.mkdir(parents=True);r=a.run_root
    command=[sys.executable,'-m','scripts.train_retarget_grounding',
        '--cache',str(r/'chain_train400_features'),'--fit-labels',str(r/'fixed_yaw_train400_s1/labels_fit.jsonl'),
        '--dev-labels',str(r/'fixed_yaw_train400_s1/labels_dev.jsonl'),'--filter',str(r/'public_deployment_filter.json'),
        '--role-vectors',str(r/'reference_vectors_v2.pt'),'--teacher',str(r/'retarget_training_data_v1'),
        '--public-inputs',str(r/'retarget_public_inputs_v1'),'--warm-start',str(r/'world_v3_envelope_s0/best.pt'),
        '--loss-mode','soft-instance']
    stages=[('smoke',['--output',str(a.output/'smoke'),'--target-language','full-instruction','--epochs','1','--steps-per-epoch','2']),
        ('resume-smoke',['--output',str(a.output/'smoke'),'--target-language','full-instruction','--epochs','2','--steps-per-epoch','2','--resume']),
        ('role',['--output',str(a.output/'role_s0'),'--target-language','role']),
        ('full-instruction',['--output',str(a.output/'full_instruction_s0'),'--target-language','full-instruction'])]
    status=dict(status='running',stages=[],started_utc=datetime.now(timezone.utc).isoformat())
    for name,extra in stages:
        status['current_stage']=name;atomic_json(status,a.output/'status.json')
        with (a.output/(name+'.log')).open('x') as stream:
            result=subprocess.run(command+extra,stdout=stream,stderr=subprocess.STDOUT)
        status['stages'].append(dict(stage=name,returncode=result.returncode,ended_utc=datetime.now(timezone.utc).isoformat()))
        if result.returncode:
            status['status']='failed';atomic_json(status,a.output/'status.json');raise SystemExit(result.returncode)
    status['status']='complete';atomic_json(status,a.output/'status.json')


if __name__=='__main__':main()
