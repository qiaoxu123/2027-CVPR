"""Train a candidate-aligned head on actual human remaining trajectories.

Static A3 is frozen and cached; labels never enter the model forward call.
Checkpoint selection uses query-family-disjoint Train Seen development loss.
"""
import argparse
import csv
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from graphnav.checkpointing import atomic_save,input_fingerprint,provenance,rng_state,restore_training
from graphnav.trajectory_data import FrozenTrajectoryDataset,move_batch
from graphnav.trajectory_policy import CandidateTrajectoryStop,trajectory_losses,calibrated_stop_probability


def run_epoch(model,loader,device,optimizer=None):
    model.train(optimizer is not None);total={};count=0;start=time.perf_counter()
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            inputs,labels=move_batch(batch,device)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.type=='cuda'):
                output=model(**inputs)
            # Metric and Loss reductions stay FP32; forward uses BF16 on the target GPU.
            output.paths_world=output.paths_world.float();output.mode_logits=output.mode_logits.float()
            output.candidate_logits=output.candidate_logits.float();output.stop_logits=output.stop_logits.float()
            world=inputs['bounds'][:,None,:2]+inputs['candidate_xy']*(inputs['bounds'][:,None,2:]-inputs['bounds'][:,None,:2])
            losses=trajectory_losses(output,candidate_world_xy=world,pose_world=inputs['pose_world'],
                **labels,stop_pos_weight=torch.tensor(20.,device=device))
            if not torch.isfinite(losses['total']):raise ValueError('non-finite trajectory training Loss')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True);losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            size=len(inputs['pose_world']);count+=size
            for key,value in losses.items():total[key]=total.get(key,0.)+float(value.detach())*size
            with torch.no_grad():
                candidate,mode,waypoint,stop=output.select();ids=torch.arange(size,device=device)
                ne=(world[ids,candidate]-labels['target_world'][:,:2]).norm(dim=-1)
                arrived=(inputs['pose_world'][:,:2]-labels['target_world'][:,:2]).norm(dim=-1)<=20.
                stopped=calibrated_stop_probability(stop,20.)>=.5
                metrics=dict(candidate_r1=(ne<=20.).float().mean(),candidate_ne=ne.mean(),
                    stop_rate=stopped.float().mean(),failed_stop_rate=(stopped & ~arrived).float().mean(),
                    missed_arrival_rate=(arrived & ~stopped).float().mean(),
                    selected_path_ade=(output.paths_world[ids,candidate,mode,:,:2]-labels['teacher_world'][...,:2]).norm(dim=-1).mean())
                for key,value in metrics.items():total[key]=total.get(key,0.)+float(value)*size
    if device.type=='cuda':torch.cuda.synchronize()
    seconds=time.perf_counter()-start
    return dict(**{key:value/max(count,1) for key,value in total.items()},samples=count,seconds=seconds,
        samples_per_second=count/max(seconds,1e-9),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if device.type=='cuda' else 0.)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('samples','queries','cache','output'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--epochs',type=int,default=5);parser.add_argument('--batch-size',type=int,default=256)
    parser.add_argument('--workers',type=int,default=4);parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--lr',type=float,default=2e-4);parser.add_argument('--device',default='cuda')
    parser.add_argument('--resume');args=parser.parse_args();out=Path(args.output)
    if out.exists() and not args.resume:raise ValueError('refusing trajectory experiment overwrite')
    out.mkdir(parents=True,exist_ok=True);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    device=torch.device(args.device)
    fingerprints=input_fingerprint(args,('samples','queries'))
    fingerprints['cache_manifest']=input_fingerprint(argparse.Namespace(manifest=str(Path(args.cache)/'manifest.json')),('manifest',))['manifest']
    train=FrozenTrajectoryDataset(args.samples,args.queries,args.cache,split='train',seed=args.seed)
    dev=FrozenTrajectoryDataset(args.samples,args.queries,args.cache,split='dev',seed=args.seed)
    # Share immutable features; DataLoader forks inherit only the compact CPU tensors.
    dev.features=train.features
    hidden=train.features['candidate_context'].shape[-1]
    model=CandidateTrajectoryStop(hidden=hidden).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr)
    source=provenance();history=[];best=float('inf');start_epoch=1
    if args.resume:
        state=restore_training(args.resume,model,optimizer,args,fingerprints)
        history=state['history'];best=state['best_loss'];start_epoch=state['epoch']+1;source=state['provenance']
    generator=torch.Generator()
    options=dict(batch_size=args.batch_size,num_workers=args.workers,persistent_workers=args.workers>0,
        pin_memory=device.type=='cuda')
    train_loader=DataLoader(train,shuffle=True,generator=generator,**options)
    dev_loader=DataLoader(dev,shuffle=False,**options)
    for epoch in range(start_epoch,args.epochs+1):
        generator.manual_seed(args.seed+epoch)
        record=dict(epoch=epoch,train=run_epoch(model,train_loader,device,optimizer),dev=run_epoch(model,dev_loader,device))
        history.append(record);improved=record['dev']['total']<best
        if improved:best=record['dev']['total']
        state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,args=vars(args),
            hidden=hidden,history=history,best_loss=best,input_fingerprint=fingerprints,provenance=source,rng=rng_state(),
            protocol=dict(observed_fraction=0.,stop_positive_weight=20.,stop_threshold=.5,
                selection='Train Seen query-family-disjoint development total loss; no Unseen selection'))
        atomic_save(state,out/'last.pt')
        if improved:atomic_save(state,out/'best.pt')
        (out/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(record),flush=True)
    rows=[dict(epoch=row['epoch'],split=split,**row[split]) for row in history for split in ('train','dev')]
    with (out/'training.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    (out/'report.json').write_text(json.dumps(dict(args=vars(args),provenance=source,input_fingerprint=fingerprints,
        training_samples=len(train),development_samples=len(dev),history=history,best_development_loss=best,
        note='Geometry/pose head training metrics, not closed-loop SR. No RGB or trained Dynamic Belief yet.'),indent=2))


if __name__=='__main__':main()
