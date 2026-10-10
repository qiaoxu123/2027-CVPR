"""Freeze trajectory/ranking; repair Stop exposure at predicted candidate endpoints."""
import argparse,csv,json,math,time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from graphnav.trajectory_data import FrozenTrajectoryDataset,move_batch
from graphnav.trajectory_policy import CandidateTrajectoryStop
from graphnav.stop_recovery import CandidateStopRecoveryDataset
from graphnav.checkpointing import provenance,input_fingerprint,atomic_save,rng_state,restore_training


def epoch(model,loader,device,optimizer=None,max_batches=None):
    model.eval();model.stop.train(optimizer is not None);count=0;loss_sum=correct=positive=stopped=true_stop=0
    start=time.perf_counter();unix=time.time();torch.cuda.reset_peak_memory_stats()
    with torch.set_grad_enabled(optimizer is not None):
        for iteration,batch in enumerate(loader):
            if max_batches and iteration>=max_batches:break
            inputs,labels=move_batch(batch,device)
            with torch.autocast('cuda',dtype=torch.bfloat16):output=model(**inputs)
            selected=output.stop_logits[torch.arange(len(labels['stop']),device=device),labels['candidate_index']].float()
            loss=F.binary_cross_entropy_with_logits(selected,labels['stop'])
            if not torch.isfinite(loss):raise ValueError('nonfinite Stop Loss')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.stop.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            size=len(selected);count+=size;loss_sum+=float(loss.detach())*size
            prediction=selected.sigmoid()>=.3;truth=labels['stop'].bool()
            correct+=int((prediction==truth).sum());positive+=int(truth.sum());stopped+=int(prediction.sum());true_stop+=int((prediction&truth).sum())
    torch.cuda.synchronize();seconds=time.perf_counter()-start
    return dict(loss=loss_sum/count,accuracy=correct/count,positive_fraction=positive/count,
        stop_precision=true_stop/max(stopped,1),stop_recall=true_stop/max(positive,1),samples=count,seconds=seconds,
        samples_per_second=count/seconds,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        started_unix=unix,finished_unix=time.time())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','samples','queries','cache','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--epochs',type=int,default=5);p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--lr',type=float,default=5e-4);p.add_argument('--resume');p.add_argument('--max-batches',type=int)
    args=p.parse_args();out=Path(args.output)
    if out.exists() and not args.resume:raise ValueError('refusing recovery Stop overwrite')
    out.mkdir(parents=True,exist_ok=True);torch.manual_seed(args.seed);device=torch.device('cuda')
    original=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    fingerprints=input_fingerprint(args,('checkpoint','samples','queries'))
    fingerprints['cache_manifest']=input_fingerprint(argparse.Namespace(manifest=str(Path(args.cache)/'manifest.json')),('manifest',))['manifest']
    train_base=FrozenTrajectoryDataset(args.samples,args.queries,args.cache,split='train',seed=0)
    dev_base=FrozenTrajectoryDataset(args.samples,args.queries,args.cache,split='dev',seed=0);dev_base.features=train_base.features
    train,dev=CandidateStopRecoveryDataset(train_base),CandidateStopRecoveryDataset(dev_base)
    model=CandidateTrajectoryStop(hidden=original['hidden']).to(device);model.load_state_dict(original['model']);model.requires_grad_(False);model.stop.requires_grad_(True)
    with torch.no_grad():model.stop[-1].bias.sub_(math.log(original['protocol']['stop_positive_weight']))
    optimizer=torch.optim.AdamW(model.stop.parameters(),lr=args.lr);source=provenance();history=[];best=float('inf');start=1
    if args.resume:
        state=restore_training(args.resume,model,optimizer,args,fingerprints);history=state['history'];best=state['best_loss'];start=state['epoch']+1;source=state['provenance']
    generator=torch.Generator();options=dict(batch_size=args.batch_size,num_workers=args.workers,persistent_workers=args.workers>0,pin_memory=True)
    train_loader=DataLoader(train,shuffle=True,generator=generator,**options);dev_loader=DataLoader(dev,shuffle=False,**options)
    for e in range(start,args.epochs+1):
        generator.manual_seed(args.seed+e)
        row=dict(epoch=e,train=epoch(model,train_loader,device,optimizer,args.max_batches),dev=epoch(model,dev_loader,device,max_batches=args.max_batches));history.append(row)
        improved=row['dev']['loss']<best
        if improved:best=row['dev']['loss']
        state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=e,args=vars(args),hidden=original['hidden'],
            input_fingerprint=fingerprints,provenance=source,rng=rng_state(),history=history,best_loss=best,
            protocol=dict(observed_fraction=0.,stop_positive_weight=1.,stop_threshold=.3,
                selection='Train query-family-disjoint endpoint Stop development BCE; no Unseen',
                label='Actual pose within released20m XY target radius; target is loss-only; candidate endpoint/jitter never uses labels'))
        atomic_save(state,out/'last.pt')
        if improved:atomic_save(state,out/'best.pt')
        (out/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(row),flush=True)
    rows=[dict(epoch=r['epoch'],split=s,**r[s]) for r in history for s in ('train','dev')]
    with (out/'training.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'report.json').write_text(json.dumps(dict(args=vars(args),provenance=source,input_fingerprint=fingerprints,
        history=history,train_samples=len(train),dev_samples=len(dev),smoke_only=bool(args.max_batches),
        note='Only Stop MLP updated. Trajectory, ranking, state fusion and A3 frozen; public predicted top8 endpoints and10m label-independent perturbations. Train internal dev includes frozen-A3-pretraining queries; primary generalization must use independent Seen closedloop.'),indent=2))


if __name__=='__main__':main()
