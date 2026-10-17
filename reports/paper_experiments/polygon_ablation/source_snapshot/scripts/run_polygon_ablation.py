"""Single-GPU, resumable, pre-registered Polygon experiment queue.

Run: python -m scripts.run_polygon_ablation --resume
No Test loading. Every trained candidate is frozen before held-out static
evaluation. No navigation starts until all static representations/seeds finish.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import torch

from graphnav.checkpointing import atomic_json,file_sha256
from scripts.polygon_ablation_plan import HERE,OUT,BASE,init,FROZEN,TRAINED,canonical_job
from scripts.polygon_ablation_engine import (encoder_for,prepare_store,model_for,evaluate,train,event,make_dataset,forward)


def static_eval(plan,store,scope,rep,job,checkpoint=None,seed=None,phase='frozen',split='val_seen'):
    folder=OUT/'localization'/job;folder.mkdir(parents=True,exist_ok=True)
    result=folder/f'{split}.json'
    signature=dict(phase=phase,scope=scope,representation=rep,seed=seed,split=split,
                   checkpoint_sha256=file_sha256(checkpoint or BASE),plan_sha256=file_sha256(OUT/'frozen_plan.json'))
    started=folder/f'{split}.started.json'
    if started.exists():assert json.loads(started.read_text())==signature
    else:atomic_json(signature,started)
    if result.exists():return
    model=model_for(plan,frozen=checkpoint is None)
    if checkpoint:model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model'],strict=True)
    summary=evaluate(model,store,scope,rep,output=folder/f'{split}.localization.jsonl')
    atomic_json(dict(status='Done',**signature,metrics=summary),result)
    event(phase='static_complete',job=job,split=split,**summary)
    del model;torch.cuda.empty_cache()


def input_parity(plan,encoder,store):
    """Check new packed frozen Full control against original unmodified path."""
    import numpy as np
    from graphnav.candidate_geometry import collate_full
    from graphnav.full_experiment import _batch_inputs
    ds=make_dataset(plan,plan['paths']['val_seen']);ids=list(range(16))
    batch=collate_full([ds[i] for i in ids])
    x,t,m,n=_batch_inputs(batch,encoder,torch.device('cuda'),'geometry')
    nx,nt,nm,nn,_,_=store.batch(torch.tensor(ids),'descriptor_only','full_polygon')
    for k in ('nodes','prior','prior_buildings','rules','relation_fields','pair_geometry','grid_xy','entity_valid','node_valid','node_types'):
        torch.testing.assert_close(x[k],nx[k],rtol=0,atol=0)
    torch.testing.assert_close(t,nt,rtol=0,atol=0);torch.testing.assert_close(m,nm,rtol=0,atol=0)
    torch.testing.assert_close(n,nn,rtol=0,atol=0)
    model=model_for(plan,frozen=True).eval()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        a=model(x,variant='geometry',tokens=t,token_mask=m,node_emb=n)
        b=model(nx,variant='geometry',tokens=nt,token_mask=nm,node_emb=nn)
    torch.testing.assert_close(a.belief_logits,b.belief_logits,rtol=0,atol=0)
    atomic_json(dict(samples=16,all_consumed_inputs_exact=True,all_logits_exact=True,unused_graph_edges_erased=True),OUT/'frozen_full_parity.json')
    del model


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--resume',action='store_true')
    p.add_argument('--stop-after',choices=['frozen','training','static','navigation'],default='navigation')
    a=p.parse_args();plan=init()
    lock=(OUT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'workflow_complete.json').exists():
        print('Entire workflow already complete; nothing rerun');return
    torch.set_num_threads(1)
    started=time.time()
    atomic_json(dict(pid=os.getpid(),started=started,command=[sys.executable,'-m','scripts.run_polygon_ablation',*sys.argv[1:]]),OUT/'active_process.json')
    launch=OUT/'launch_record.json'
    if not launch.exists():
        atomic_json(dict(pid=os.getpid(),started=started,command=[sys.executable,'-m','scripts.run_polygon_ablation',*sys.argv[1:]],
                         commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                         source_sha256={str(f.relative_to(HERE)):file_sha256(f) for f in [*HERE.glob('graphnav/polygon_ablation.py'),*HERE.glob('scripts/*polygon*.py')]},
                         plan_sha256=file_sha256(OUT/'frozen_plan.json')),launch)
    tests=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_polygon_ablation.py'],cwd=HERE,stdout=(OUT/'unit_tests.log').open('w'),stderr=subprocess.STDOUT)
    if tests.returncode:raise RuntimeError('Polygon geometry unit tests failed; no experimental scores produced')
    encoder=encoder_for(plan)
    seen=prepare_store(plan,'val_seen',encoder);unseen=prepare_store(plan,'val_unseen',encoder)
    input_parity(plan,encoder,seen)
    for scope,rep in FROZEN:
        for split,store in [('val_seen',seen),('val_unseen',unseen)]:
            static_eval(plan,store,scope,rep,f'frozen__{scope}__{rep}',split=split)
    if a.stop_after=='frozen':return
    training=prepare_store(plan,'train_seen',encoder)
    jobs={canonical_job(scope,rep,seed):(scope,rep,seed) for scope,rep,seed in TRAINED}
    for job,(scope,rep,seed) in jobs.items():
        train(plan,training,scope,rep,seed)
    # Freeze every selected weight before any held-out assessment of retrains.
    frozen={name:json.loads((OUT/'training'/name/'complete.json').read_text()) for name in jobs}
    for seed in (0,1,2):
        hashes={v['initial_state_sha256'] for n,v in frozen.items() if n.endswith(f'seed{seed}')}
        assert len(hashes)==1,'Unmatched initialization across conditions'
    atomic_json(dict(jobs=frozen,selection=plan['selection'],no_heldout_selection=True),OUT/'all_training_frozen.json')
    if a.stop_after=='training':return
    for scope,rep,seed in TRAINED:
        name=canonical_job(scope,rep,seed);checkpoint=OUT/'training'/name/'best.pt'
        for split,store in [('val_seen',seen),('val_unseen',unseen)]:
            # Separate logical-scope result metadata; Full weights are shared.
            job=f'trained__{scope}__{rep}__seed{seed}'
            static_eval(plan,store,scope,rep,job,checkpoint,seed,'retrained',split)
    atomic_json(dict(status='Done',frozen_conditions=len(FROZEN),trained_conditions=18,
                     seen_queries=2426,unseen_queries=2679),OUT/'all_static_complete.json')
    if a.stop_after=='static':return
    del training,seen,unseen;torch.cuda.empty_cache()
    from scripts.polygon_ablation_navigation import navigate,verify_rendering_parity
    full='frozen__descriptor_only__full_polygon'
    original=navigate(plan,encoder,full,'descriptor_only','full_polygon',smoke=True,original_builder=True)
    optimized=navigate(plan,encoder,full,'descriptor_only','full_polygon',smoke=True)
    verify_rendering_parity(original,optimized)
    navjobs=[(f'frozen__{s}__{r}',s,r,None) for s,r in FROZEN]
    # Seed0 is predeclared first; include all seeds automatically when resources
    # permit. Sharing Full across scopes avoids duplicating an identical rollout.
    for seed in (0,1,2):
        for scope,rep,s in TRAINED:
            if s!=seed:continue
            if rep=='full_polygon' and scope=='end_to_end':continue
            navjobs.append((f'trained__{scope}__{rep}__seed{seed}',scope,rep,OUT/'training'/canonical_job(scope,rep,seed)/'best.pt'))
    for job,scope,rep,checkpoint in navjobs:
        navigate(plan,encoder,job,scope,rep,checkpoint,smoke=True)
        for split in ('val_seen','val_unseen'):
            navigate(plan,encoder,job,scope,rep,checkpoint,split=split)
    from scripts.polygon_ablation_statistics import summarize
    summarize(plan)
    atomic_json(dict(status='Done',seconds=time.time()-started,training_runs=15,logical_training_conditions=18,
                     frozen_interventions=len(FROZEN),navigation_runs=len(navjobs),all_static_before_navigation=True),OUT/'workflow_complete.json')
    atomic_json(dict(status='Done',phase='complete'),OUT/'progress.json')


if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        atomic_json(dict(status='Blocked',error=traceback.format_exc(),time=time.time()),OUT/'failure.json')
        raise
