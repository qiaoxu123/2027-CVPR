"""Verify completed core pair and publish small reports; never publish large runs."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
import torch
from graphnav.checkpointing import atomic_json,file_sha256

ROOT=Path('runs/joint_rgb_geometry_core_20261027')
REPORT=Path('reports/core_pair_20261027')


def command(*args):return subprocess.check_output(args,text=True).strip()


def main():
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');p.add_argument('--a1-only',action='store_true');a=p.parse_args()
    global ROOT
    if a.a1_only:ROOT=Path('runs/joint_rgb_geometry_A1_20261027')
    experiment_unit='graphnav-rgb-A1-20261027' if a.a1_only else 'graphnav-corepair-20261027'
    while True:
        state=command('systemctl','--user','show',experiment_unit,'--property=ActiveState','--value')
        if state not in ('active','activating'):break
        if not a.wait:raise RuntimeError('training/evaluation still running')
        time.sleep(30)
    exitcode=command('systemctl','--user','show',experiment_unit,'--property=ExecMainStatus','--value')
    if exitcode!='0' or state=='failed':raise RuntimeError('core pair failed; refusing final publication')
    result=json.loads((ROOT/'core_pair_results.json').read_text());frozen=json.loads((ROOT/'frozen_pair.json').read_text())
    inventory={};frozen_dev=None
    arms=('A1_joint_rgb_geometry',)if a.a1_only else ('A0_original_residual','A1_joint_rgb_geometry')
    for arm in arms:
        inventory[arm]={}
        for stage,epochs in (('A',3),('B',3),('C',5)):
            folder=ROOT/arm/'stages'/stage;done=json.loads((folder/'complete.json').read_text());history=json.loads((folder/'history.json').read_text())
            if done['epochs']!=epochs or done['smoke'] or len(history)!=epochs:raise ValueError('incomplete stage')
            if any(r['train']['samples']!=83140 or r['dev']['samples']!=4372 for r in history):raise ValueError('partial train/dev epoch')
            checkpoint=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
            dev=checkpoint['config']['dev_episodes']
            if frozen_dev is None:frozen_dev=dev
            if dev!=frozen_dev:raise ValueError('internal dev split differs')
            if checkpoint['args']['seed']!=0 or checkpoint['args']['batch_size']!=16:raise ValueError('training config differs')
            events=list((folder/'tensorboard').glob('events.out.tfevents.*'))
            if not events or any(e.stat().st_size==0 for e in events):raise ValueError('missing TensorBoard curves')
            inventory[arm][stage]=dict(epochs=epochs,train_states=83140,dev_states=4372,best_checkpoint_sha256=file_sha256(folder/'best.pt'),last_checkpoint_sha256=file_sha256(folder/'last.pt'),tensorboard_events=[str(e.resolve())for e in events],train_seconds=sum(r['train']['seconds']for r in history),history_sha256=file_sha256(folder/'history.json'),training_commits=sorted({checkpoint['provenance']['commit']}))
        for split,n in (('val_seen',2470),('val_unseen',2697)):
            report=json.loads((ROOT/arm/('full_'+split)/'metrics.json').read_text())
            if report['metrics']['episodes']!=n or report['checkpoint_sha256']!=frozen['arms'][arm]['checkpoint_sha256']:raise ValueError('final evaluation count or frozen SHA differs')
            if file_sha256(frozen['arms'][arm]['checkpoint'])!=frozen['arms'][arm]['checkpoint_sha256']:raise ValueError('frozen checkpoint modified')
    atomic_json(dict(complete=True,verified_at_utc=datetime.now(timezone.utc).isoformat(),server_root=str(ROOT.resolve()),arms=inventory),REPORT/'training_inventory.json')
    for name in ('frozen_pair.json','paired_protocol.json'):(REPORT/name).write_bytes((ROOT/name).read_bytes())
    plan=Path('docs/superpowers/plans/2026-10-27-core-pair.md');plan.write_text(plan.read_text().replace('- [ ]','- [x]'))
    # Exclude this last checklist item until the remote update has succeeded.
    branch=command('git','branch','--show-current')
    if branch!='research/joint-rgb-geometry-retrain-20261027':raise ValueError('wrong publication branch')
    subprocess.run(['git','diff','--check'],check=True)
    subprocess.run(['git','add','reports/core_pair_20261027','docs/superpowers/plans/2026-10-27-core-pair.md','scripts/finalize_core_pair.py'],check=True)
    subprocess.run(['git','commit','-m','Report completed RGB fusion training and full validation results'],check=True)
    subprocess.run(['git','push','origin','HEAD:'+branch],check=True)
    head=command('git','rev-parse','HEAD');remote=command('git','ls-remote','origin','refs/heads/'+branch).split()[0]
    if head!=remote:raise ValueError('remote commit verification failed')
    atomic_json(dict(status='complete',commit=head,remote_verified=True,completed_at_utc=datetime.now(timezone.utc).isoformat()),ROOT/'publication_complete.json')
    print(json.dumps(dict(status='complete',commit=head,results=str((REPORT/'RESULTS.md').resolve()))),flush=True)

if __name__=='__main__':main()
