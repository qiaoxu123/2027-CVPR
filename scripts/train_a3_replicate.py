"""Seen-selected A3 replication with persistent frozen BERT, never Unseen tuning."""
import argparse,csv,json,random,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader
from graphnav.candidate_geometry import FullCandidateDataset,collate_full
from graphnav.full_experiment import new_model,_batch_inputs
from graphnav.token_language import FrozenBERTTokens
from graphnav.static_localization import cross_entropy_goal,static_recall
from graphnav.checkpointing import atomic_save,provenance,input_fingerprint,rng_state,restore_training


def epoch(model,loader,encoder,device,optimizer=None,max_batches=None):
    model.train(optimizer is not None);started=time.perf_counter();unix=time.time();torch.cuda.reset_peak_memory_stats()
    total={'loss':0.,'NE_sum_m':0.,'R@1/20m_count':0.,'R@4/20m_count':0.,'R@8/20m_count':0.};count=0
    with torch.set_grad_enabled(optimizer is not None):
        for iteration,batch in enumerate(loader):
            if max_batches and iteration>=max_batches:break
            inputs,tokens,mask,names=_batch_inputs(batch,encoder,device,'geometry')
            with torch.autocast('cuda',dtype=torch.bfloat16):field=model(inputs,variant='geometry',tokens=tokens,token_mask=mask,node_emb=names)
            target=batch['target_xy'].to(device);loss=cross_entropy_goal(field.belief_logits,target,grid_size=model.grid_size if hasattr(model,'grid_size') else int(len(inputs['grid_xy'][0])**.5))
            if not torch.isfinite(loss):raise ValueError('nonfinite A3 Loss')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            size=len(target);count+=size;total['loss']+=float(loss.detach())*size
            with torch.no_grad():
                metrics=static_recall(field.belief_logits.float(),target,batch['map_sizes'].to(device),inputs['grid_xy'])
                for key in total:
                    if key!='loss':total[key]+=metrics[key]
    torch.cuda.synchronize();seconds=time.perf_counter()-started
    return dict(loss=total['loss']/count,NE_m=total['NE_sum_m']/count,
        **{f'R@{k}/20m':total[f'R@{k}/20m_count']/count for k in (1,4,8)},samples=count,
        seconds=seconds,samples_per_second=count/seconds,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        started_unix=unix,finished_unix=time.time())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('template','train','seen','train-bert-cache','seen-bert-cache','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--epochs',type=int,default=5);p.add_argument('--resume')
    p.add_argument('--max-batches',type=int,help='Smoke only; limited runs are ineligible final candidates')
    p.add_argument('--init-bidirectional',help='Only difference from a paired Scratch replicate: verified P1 initialization')
    args=p.parse_args();out=Path(args.output)
    if out.exists() and not args.resume:raise ValueError('refusing A3 replicate overwrite')
    out.mkdir(parents=True,exist_ok=True);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    template=torch.load(args.template,map_location='cpu',weights_only=True)
    cfg=SimpleNamespace(**dict(template['args'],seed=args.seed,batch_size=args.batch_size,workers=args.workers,
        epochs=args.epochs,train=args.train,val_seen=args.seen,output=args.output,max_text_length=96))
    cfg.resume=args.resume;cfg.train_bert_cache=args.train_bert_cache;cfg.seen_bert_cache=args.seen_bert_cache
    cfg.max_train_batches=args.max_batches or 0
    fingerprint=input_fingerprint(args,('template','train','seen','train_bert_cache','seen_bert_cache','init_bidirectional') if args.init_bidirectional else ('template','train','seen','train_bert_cache','seen_bert_cache'))
    cfg.init_bidirectional=args.init_bidirectional
    device=torch.device('cuda');encoder=FrozenBERTTokens(cfg.text_backbone,device='cuda',max_length=96)
    encoder.load_cache(args.train_bert_cache);training_cache=encoder.cache.copy();encoder.load_cache(args.seen_bert_cache);encoder.cache.update(training_cache);del training_cache
    def dataset(path):return FullCandidateDataset(path,grid_size=cfg.grid_size,max_landmarks=cfg.max_landmarks,
        max_roads=cfg.max_roads,max_relations=cfg.max_relations,prior_sigma_m=cfg.prior_sigma_m,rule_sigma_m=cfg.rule_sigma_m)
    train=dataset(args.train);seen=dataset(args.seen)
    if {(r['map_name'],r['instruction']) for r in train.records}&{(r['map_name'],r['instruction']) for r in seen.records}:raise ValueError('train/Seen query leakage')
    # Reset after frozen model construction: trainable initialization depends only on seed.
    torch.manual_seed(args.seed);model=new_model(cfg,device)
    if args.init_bidirectional:
        initial=torch.load(args.init_bidirectional,map_location='cpu',weights_only=True)
        if initial.get('training_stage')!='P1_bidirectional_geometry':raise ValueError('verified P1 checkpoint required')
        if initial['args'].get('max_batches'):raise ValueError('limited P1 smoke is not a valid initialization')
        if (initial['hidden'],initial['heads'],initial['grid_size'],initial['relative_position'])!=(cfg.hidden,cfg.heads,cfg.grid_size,getattr(cfg,'relative_position',False)):
            raise ValueError('P1 and A3 architecture differ')
        model.load_state_dict(initial['model'],strict=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.lr)
    generator=torch.Generator();options=dict(batch_size=cfg.batch_size,num_workers=cfg.workers,
        persistent_workers=cfg.workers>0,pin_memory=True,collate_fn=collate_full)
    train_loader=DataLoader(train,shuffle=True,generator=generator,**options);seen_loader=DataLoader(seen,shuffle=False,**options)
    source=provenance();history=[];best=(-1.,float('-inf'));start=1
    if args.resume:
        state=restore_training(args.resume,model,optimizer,cfg,fingerprint);history=state['history'];best=tuple(state['best_key']);start=state['epoch']+1;source=state['provenance']
    for e in range(start,args.epochs+1):
        generator.manual_seed(args.seed+e)
        row=dict(epoch=e,train=epoch(model,train_loader,encoder,device,optimizer,args.max_batches),seen=epoch(model,seen_loader,encoder,device,max_batches=args.max_batches))
        history.append(row);key=(row['seen']['R@1/20m'],-row['seen']['NE_m']);improved=key>best
        if improved:best=key
        state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=e,args=vars(cfg),variant='geometry',
            dataset_version='revised',match_mode='fuzzy',selected_on='Val Seen R@1/20m then NE; Unseen untouched',
            best_key=best,history=history,input_fingerprint=fingerprint,provenance=source,rng=rng_state())
        atomic_save(state,out/'last.pt')
        if improved:atomic_save(state,out/'best.pt')
        (out/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(row),flush=True)
    rows=[dict(epoch=r['epoch'],split=s,**r[s]) for r in history for s in ('train','seen')]
    with (out/'training.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'report.json').write_text(json.dumps(dict(args=vars(cfg),provenance=source,input_fingerprint=fingerprint,history=history,
        smoke_only=bool(args.max_batches),note='Legacy A3 structure/optimizer, explicit seed and persistent BERT96. Scratch initialization; best by Seen R1 thenNE; no Unseen loaded or evaluated. Limited smoke is ineligible final candidate.'),indent=2))


if __name__=='__main__':main()
