"""Persistent, matched re-pairing coarse/soft-instance TRAIN pilots."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import sys
from graphnav.checkpointing import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve previous suite')
    args.output.mkdir(parents=True);root=args.run_root
    command=[sys.executable,'-m','scripts.train_retarget_grounding',
        '--cache',str(root/'chain_train400_features'),'--fit-labels',str(root/'fixed_yaw_train400_s1/labels_fit.jsonl'),
        '--dev-labels',str(root/'fixed_yaw_train400_s1/labels_dev.jsonl'),'--filter',str(root/'public_deployment_filter.json'),
        '--role-vectors',str(root/'reference_vectors_v2.pt'),'--teacher',str(root/'retarget_training_data_v1'),
        '--public-inputs',str(root/'retarget_public_inputs_v1'),'--warm-start',str(root/'world_v3_envelope_s0/best.pt')]
    stages=[('smoke',['--output',str(args.output/'smoke'),'--loss-mode','soft-instance','--epochs','1','--steps-per-epoch','2']),
        ('resume-smoke',['--output',str(args.output/'smoke'),'--loss-mode','soft-instance','--epochs','2','--steps-per-epoch','2','--resume']),
        ('coarse',['--output',str(args.output/'coarse_s0'),'--loss-mode','coarse']),
        ('soft-instance',['--output',str(args.output/'soft_instance_s0'),'--loss-mode','soft-instance'])]
    status=dict(status='running',stages=[],started_utc=datetime.now(timezone.utc).isoformat())
    for name,extra in stages:
        status['current_stage']=name;atomic_json(status,args.output/'status.json')
        with (args.output/(name+'.log')).open('x') as stream:result=subprocess.run(command+extra,stdout=stream,stderr=subprocess.STDOUT)
        status['stages'].append(dict(stage=name,returncode=result.returncode,ended_utc=datetime.now(timezone.utc).isoformat()))
        if result.returncode:
            status['status']='failed';atomic_json(status,args.output/'status.json');raise SystemExit(result.returncode)
    status['status']='complete';atomic_json(status,args.output/'status.json')


if __name__=='__main__':main()
