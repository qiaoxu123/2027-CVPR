"""Train bounded A3-preserving graph/visual residuals on causal real observations."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random
import time
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader
from graphnav.checkpointing import atomic_save,file_sha256,provenance,rng_state,restore_training
from graphnav.dynamic_data import DynamicEvidenceDataset
from graphnav.dynamic_belief import DynamicGoalBelief
from graphnav.full_localization import GoalField
from graphnav.full_experiment import new_model


def forward_cached(model,inputs,args):
    logits=inputs['base_logits'];dense=inputs['dense_context']
    base=GoalField(logits,None,None,None,dense_context=dense)
    fields={key:value for key,value in inputs.items() if key in ('nodes','node_types','node_valid','entity_valid','adjacency','edges','pair_geometry','grid_xy')}
    return model(fields,tokens=None,token_mask=None,node_emb=inputs['node_emb'],base_field=base,
        use_graph=args.variant!='visual',use_visual=args.variant!='graph',
        visual_vectors=inputs.get('visual_vectors'),visual_confidence=inputs.get('visual_confidence'),
        observed_vectors=inputs['observed_vectors'],observed_confidence=inputs['observed_confidence'],observed_geometry=inputs['observed_geometry'])


def state_metrics(logits,inputs,labels):
    ids=logits.float().topk(8,-1).indices
    xy=inputs['grid_xy'].gather(1,ids[...,None].expand(-1,-1,2))
    distances=((xy-labels['target_xy'][:,None])*labels['map_sizes'][:,None]).norm(dim=-1)
    return dict(NE=distances[:,0],R1=(distances[:,:1].amin(-1)<=20).float(),
        R4=(distances[:,:4].amin(-1)<=20).float(),R8=(distances.amin(-1)<=20).float())


def run_epoch(model,loader,args,device,optimizer=None,prediction_path=None):
    model.train(optimizer is not None);start=time.perf_counter();started_utc=time.time();count=0;total=0.;groups=defaultdict(list);predictions=[]
    reference_observations,auxiliary_observations,reference_visible_states=0,0,0
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    with torch.set_grad_enabled(optimizer is not None):
        for batch_index,(inputs,labels) in enumerate(loader):
            if getattr(args,'max_batches',None) is not None and batch_index>=args.max_batches:break
            inputs={key:value.to(device,non_blocking=True) for key,value in inputs.items()}
            labels={key:value.to(device,non_blocking=True) for key,value in labels.items()}
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.type=='cuda'):output=forward_cached(model,inputs,args)
            logits=output.belief_logits.float();base=inputs['base_logits'].float()
            distance=((inputs['grid_xy']-labels['target_xy'][:,None])*labels['map_sizes'][:,None]).norm(dim=-1)
            soft_target=(-distance.square()/(2*20.**2)).softmax(-1)
            loss=-(soft_target*logits.log_softmax(-1)).sum(-1).mean()
            # Preserve the strongest prior unless actual supervised evidence improves it.
            preservation=torch.nn.functional.kl_div(logits.log_softmax(-1),base.softmax(-1),reduction='batchmean')
            loss=loss+.05*preservation
            if not torch.isfinite(loss):raise ValueError('nonfinite dynamic-belief Loss')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            size=len(logits);count+=size;total+=float(loss.detach())*size
            if 'visual_confidence' in inputs:
                reference_observations+=int((inputs['visual_confidence']>0).sum())
                reference_visible_states+=int((inputs['visual_confidence']>0).any(-1).sum())
            auxiliary_observations+=int((inputs['observed_confidence']>0).sum())
            with torch.no_grad():
                values=state_metrics(logits,inputs,labels);baseline=state_metrics(base,inputs,labels)
                qids=labels['query_index'].cpu().tolist();steps=labels['source_step'].cpu().tolist()
                metrics={key:value.cpu().tolist() for key,value in values.items()}
                original={key:value.cpu().tolist() for key,value in baseline.items()}
                for position,qid in enumerate(qids):
                    row=dict(query_index=qid,source_step=steps[position],
                        **{key:value[position] for key,value in metrics.items()},
                        **{'baseline_'+key:value[position] for key,value in original.items()})
                    groups[qid].append(row)
                    if prediction_path is not None:predictions.append(row)
    if device.type=='cuda':torch.cuda.synchronize()
    seconds=time.perf_counter()-start
    # Every public query gets equal mass despite different available prefix counts.
    names=('NE','R1','R4','R8','baseline_NE','baseline_R1','baseline_R4','baseline_R8')
    macro={name:float(np.mean([np.mean([row[name] for row in rows]) for rows in groups.values()])) for name in names}
    if prediction_path is not None:Path(prediction_path).write_text(''.join(json.dumps(row)+'\n' for row in predictions))
    return dict(loss=total/max(count,1),state_queries=count,unique_public_queries=len(groups),query_macro=macro,
        seconds=seconds,started_unix=started_utc,finished_unix=time.time(),samples_per_second=count/max(seconds,1e-9),
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if device.type=='cuda' else 0.,
        graph_gate=float(torch.tanh(model.graph_gate).detach()),visual_gate=float(torch.tanh(model.visual_gate).detach()),
        reference_observations=reference_observations,auxiliary_observations=auxiliary_observations,
        reference_visible_state_fraction=reference_visible_states/max(count,1))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('train-evidence','train-queries','train-fields','train-vectors','seen-evidence','seen-queries','seen-fields','seen-vectors','checkpoint','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--variant',choices=['graph','visual','graph_visual_history'],required=True)
    p.add_argument('--associate-observed-refs',action='store_true',help='Bind observed public contours/names to GAT reference nodes before message passing')
    p.add_argument('--epochs',type=int,default=5);p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--seed',type=int,default=0);p.add_argument('--split-seed',type=int,default=0)
    p.add_argument('--lr',type=float,default=2e-4);p.add_argument('--gate-lr',type=float,default=2e-3)
    p.add_argument('--max-batches',type=int,help='Smoke only: explicitly truncate each split; never use as full evaluation')
    p.add_argument('--resume');p.add_argument('--device',default='cuda');args=p.parse_args();out=Path(args.output)
    if out.exists() and not args.resume:raise ValueError('refusing dynamic experiment overwrite')
    out.mkdir(parents=True,exist_ok=True);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True);config=SimpleNamespace(**state['args']);device=torch.device(args.device)
    a3=new_model(config,device);a3.load_state_dict(state['model'],strict=True)
    model=DynamicGoalBelief(a3,vision_dim=256).to(device)
    def dataset(prefix,split):
        return DynamicEvidenceDataset(getattr(args,prefix+'_evidence'),getattr(args,prefix+'_queries'),getattr(args,prefix+'_fields'),getattr(args,prefix+'_vectors'),config,
            history=args.variant=='graph_visual_history',split=split,split_seed=args.split_seed,associate_references=args.associate_observed_refs)
    train=dataset('train','train');dev=dataset('train','dev');seen=dataset('seen','all')
    # The datasets are compact query-subset feature stores; frozen encoders are absent.
    fingerprint={key:dict(path=str(Path(value).resolve()),sha256=file_sha256(value)) for key,value in vars(args).items() if key.endswith('_queries') or key=='checkpoint'}
    for key in ('train_evidence','train_fields','train_vectors','seen_evidence','seen_fields','seen_vectors'):
        path=Path(getattr(args,key))/'manifest.json';fingerprint[key]=dict(path=str(path.resolve()),sha256=file_sha256(path))
    optimizer=torch.optim.AdamW([dict(params=[parameter for name,parameter in model.named_parameters() if parameter.requires_grad and not name.endswith('_gate')],lr=args.lr),
        dict(params=[model.graph_gate,model.visual_gate],lr=args.gate_lr)])
    source=provenance();history=[];best_key=(-float('inf'),-float('inf'));start_epoch=1
    if args.resume:
        saved=restore_training(args.resume,model,optimizer,args,fingerprint);history=saved['history'];best_key=tuple(saved['best_key']);start_epoch=saved['epoch']+1;source=saved['provenance']
    generator=torch.Generator();options=dict(batch_size=args.batch_size,num_workers=args.workers,persistent_workers=args.workers>0,pin_memory=device.type=='cuda')
    train_loader=DataLoader(train,shuffle=True,generator=generator,**options);dev_loader=DataLoader(dev,shuffle=False,**options);seen_loader=DataLoader(seen,shuffle=False,**options)
    for epoch in range(start_epoch,args.epochs+1):
        generator.manual_seed(args.seed+epoch)
        record=dict(epoch=epoch,train=run_epoch(model,train_loader,args,device,optimizer),dev=run_epoch(model,dev_loader,args,device))
        history.append(record);key=(record['dev']['query_macro']['R1'],-record['dev']['query_macro']['NE']);improved=key>best_key
        if improved:best_key=key
        saved=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),args=vars(args),epoch=epoch,best_key=best_key,
            history=history,input_fingerprint=fingerprint,provenance=source,rng=rng_state(),
            protocol='FP32 frozen A3 logits; FP16 frozen dense contexts; causal current/past actual ROI vectors; query-family-disjoint train dev checkpoint choice, no Unseen.')
        atomic_save(saved,out/'last.pt')
        if improved:atomic_save(saved,out/'best.pt')
        (out/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(record),flush=True)
    best=torch.load(out/'best.pt',map_location=device,weights_only=True);model.load_state_dict(best['model'],strict=True)
    result=run_epoch(model,seen_loader,args,device,prediction_path=out/'seen_prefix_predictions.jsonl')
    report=dict(args=vars(args),provenance=source,input_fingerprint=fingerprint,best_epoch=best['epoch'],history=history,seen_prefix=result,
        note='Seen actual-human-prefix subset query-macro localization diagnostic; not full navigation SR/SPL or initial-only observations. No Unseen evaluation.')
    (out/'report.json').write_text(json.dumps(report,indent=2));rows=[dict(epoch=row['epoch'],split=split,**{key:value for key,value in row[split].items() if key!='query_macro'},**row[split]['query_macro']) for row in history for split in ('train','dev')]
    with (out/'training.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
