"""Persistent fixed-budget Local/Global/Constant reference evidence comparison."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import sys
from graphnav.checkpointing import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve old suite')
    a.output.mkdir(parents=True)
    command=[sys.executable,'-m','scripts.train_reference_evidence_transport','--root',str(a.root)]
    stages=[('smoke',['--output',str(a.output/'smoke'),'--mode','global','--epochs','1','--steps-per-epoch','2']),
            ('resume-smoke',['--output',str(a.output/'smoke'),'--mode','global','--epochs','2','--steps-per-epoch','2','--resume'])]
    stages += [(m,['--output',str(a.output/(m+'_s0')),'--mode',m]) for m in ('local','global','constant')]
    status=dict(status='running',stages=[],started_utc=datetime.now(timezone.utc).isoformat())
    for name,extra in stages:
        status['current_stage']=name;atomic_json(status,a.output/'status.json')
        with (a.output/(name+'.log')).open('x') as stream:result=subprocess.run(command+extra,stdout=stream,stderr=subprocess.STDOUT)
        status['stages'].append(dict(stage=name,returncode=result.returncode,ended_utc=datetime.now(timezone.utc).isoformat()))
        if result.returncode:
            status['status']='failed';atomic_json(status,a.output/'status.json');raise SystemExit(result.returncode)
    status['status']='complete';atomic_json(status,a.output/'status.json')


if __name__=='__main__':main()
