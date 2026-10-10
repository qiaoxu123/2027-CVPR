"""Three-epoch matched pilot; only lightweight heads see gradients.

A3/SigLIP inference is outside this process, in the frozen cache builder.
No Seen model selection: the completed epoch-3 checkpoint is always final.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from graphnav.checkpointing import atomic_save, atomic_json, file_sha256, rng_state
from .model import CoreModel, collate_core, core_losses


def tensor_state_sha(state):
    h=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        h.update(name.encode());h.update(str(tensor.shape).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def verify_frozen_resources(manifest):
    result={}
    for path, expected in manifest.get('frozen_resources',{}).items():
        actual=file_sha256(path)
        if actual!=expected: raise ValueError('frozen resource changed: '+path)
        result[path]=actual
    return result


def train(args):
    from .cache import CachedCoreDataset,finalize_cache_integrity
    out=Path(args.output)
    if out.exists(): raise ValueError('refusing to overwrite core training output')
    cache=Path(args.cache);manifest=finalize_cache_integrity(cache)
    if not manifest.get('complete',False):raise ValueError('incomplete frozen feature cache')
    resources=verify_frozen_resources(manifest)
    data=CachedCoreDataset(cache,condition=args.condition)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    device=torch.device(args.device)
    if device.type=='cuda':torch.cuda.manual_seed_all(args.seed)
    hidden=int(manifest.get('hidden',128));top_k=int(manifest.get('top_k',20))
    model=CoreModel(hidden=hidden,top_k=top_k).to(device)
    init_sha=tensor_state_sha(model.state_dict())
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    steps=math.ceil(len(data)/args.batch_size)*args.epochs
    warmup=max(1,int(.05*steps))
    def factor(step):
        if step<warmup:return (step+1)/warmup
        return .5*(1+math.cos(math.pi*min(1,(step-warmup)/max(1,steps-warmup))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,factor)
    out.mkdir(parents=True)
    protocol=dict(name='bidirectional-core-v1',condition=args.condition,
        freeze_a3=True,freeze_siglip=True,freeze_bert=True,no_target_in_forward=True,
        visual_language_supervision='conservative target identity multi-positive, not true attribute labels',
        final_selection='fixed completed epoch3; no Seen best selection',
        static_cache=True,steps_per_epoch=math.ceil(len(data)/args.batch_size),
        samples_per_epoch=len(data),epochs=args.epochs,batch_size=args.batch_size,
        seed=args.seed,initial_head_sha256=init_sha,cache_manifest_sha256=file_sha256(cache/'manifest.json'),
        frozen_resources=resources,trainable_parameters=sum(p.numel() for p in model.parameters()),
        hidden=hidden,top_k=top_k,image_precision=manifest.get('image_precision'),vl_weight=args.vl_weight if args.condition=='E2' else 0.,
        stop_positive_weight=20.,horizontal_step_m=50.,vertical_step_m=10.)
    atomic_json(protocol,out/'protocol.json')
    history=[];global_step=0
    for epoch in range(1,args.epochs+1):
        generator=torch.Generator().manual_seed(args.seed+epoch)
        loader=DataLoader(data,batch_size=args.batch_size,shuffle=True,generator=generator,
            collate_fn=collate_core,num_workers=args.workers,pin_memory=device.type=='cuda')
        model.train();totals={};count=0;start=time.perf_counter();first_grad=None;matcher_gradient_seen=False
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        for batch_index,(public,labels,metadata) in enumerate(loader):
            public={k:v.to(device,non_blocking=True) for k,v in public.items()}
            labels={k:v.to(device,non_blocking=True) for k,v in labels.items()}
            if any(v.requires_grad for v in public.values()):raise ValueError('cached input unexpectedly requires gradient')
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                output=model(public)
            losses=core_losses(output,public,labels,vl_weight=protocol['vl_weight'])
            if not torch.isfinite(losses['total']):raise ValueError('nonfinite matched pilot loss')
            losses['total'].backward()
            matcher_gradient_seen = matcher_gradient_seen or any(p.grad is not None and bool(p.grad.abs().max()>0) for p in model.matcher.parameters())
            if first_grad is None:
                first_grad={name:sum(float(p.grad.float().norm()) for p in module.parameters() if p.grad is not None)
                    for name,module in [('matcher',model.matcher),('trajectory_stop',model.trajectory)]}
                first_grad['visual_gain']=float(model.visual_gain.grad)
                if first_grad['trajectory_stop']<=0:raise ValueError('missing trajectory gradient')
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();scheduler.step();global_step+=1
            n=len(public['pose_world']);count+=n
            for key,value in losses.items():
                totals[key]=totals.get(key,0.)+float(value.detach())*(1 if key=='visual_language_eligible' else n)
            if (batch_index+1)%100==0:
                print(json.dumps(dict(condition=args.condition,epoch=epoch,batch=batch_index+1,
                    samples=count,seconds=time.perf_counter()-start)),flush=True)
        if device.type=='cuda':torch.cuda.synchronize(device)
        frozen=verify_frozen_resources(manifest)
        if file_sha256(cache/'manifest.json') != protocol['cache_manifest_sha256']:
            raise ValueError('frozen cache manifest changed during training')
        record=dict(epoch=epoch,global_step=global_step,samples=count,seconds=time.perf_counter()-start,
            losses={k:v/(1 if k=='visual_language_eligible' else count) for k,v in totals.items()},
            first_batch_gradients=first_grad,matcher_gradient_observed=matcher_gradient_seen,frozen_resources_unchanged=frozen==resources,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)
        history.append(record)
        state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            epoch=epoch,args=vars(args),protocol=protocol,history=history,rng=rng_state())
        atomic_save(state,out/f'epoch{epoch:02}.pt');atomic_save(state,out/'last.pt')
        atomic_json(history,out/'history.json');print(json.dumps(record),flush=True)
    atomic_json(dict(completed_epochs=args.epochs,global_steps=global_step,protocol=protocol,
        final_checkpoint=str(out/f'epoch{args.epochs:02}.pt'),final_checkpoint_sha256=file_sha256(out/f'epoch{args.epochs:02}.pt')),
        out/'training_complete.json')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',required=True);p.add_argument('--output',required=True)
    p.add_argument('--condition',choices=['E0','E1','E2'],required=True)
    p.add_argument('--epochs',type=int,default=3);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--vl-weight',type=float,default=.1);p.add_argument('--workers',type=int,default=0)
    p.add_argument('--device',default='cuda');args=p.parse_args()
    if args.epochs!=3:raise ValueError('fixed three-epoch matched pilot')
    train(args)

if __name__=='__main__':main()
