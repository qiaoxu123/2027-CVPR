"""Train one complete joint model; select on Seen closed loop, never on Unseen."""
import argparse
import csv
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from graphnav.checkpointing import atomic_save,file_sha256,provenance,rng_state,restore_training
from graphnav.full_experiment import _batch_inputs,new_model
from graphnav.joint_active import JointActiveGraphNav,joint_losses
from graphnav.joint_active_data import JointActiveDataset,collate_joint,ShardShuffleSampler
from graphnav.token_language import FrozenBERTTokens


def run_epoch(model,loader,bert,device,optimizer,scheduler,*,max_batches=None):
    model.train();total={};count=0;began=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    first_grad=None;gpu_utilization=[];visual_gradient_seen=False
    for index,(static,public,labels) in enumerate(loader):
        if max_batches is not None and index>=max_batches:break
        static,tokens,mask,node_emb=_batch_inputs(static,bert,device,'geometry')
        public={k:v.to(device,non_blocking=True) for k,v in public.items()}
        labels={k:v.to(device,non_blocking=True) for k,v in labels.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            output=model(static_inputs=static,tokens=tokens,token_mask=mask,node_emb=node_emb,**public)
        grid_world=public['bounds'][:,None,:2]+static['grid_xy']*(public['bounds'][:,None,2:]-public['bounds'][:,None,:2])
        loss=joint_losses(output,grid_world=grid_world,pose_world=public['pose_world'],**labels)
        if not torch.isfinite(loss['total']):raise ValueError('nonfinite complete-model loss')
        loss['total'].backward()
        if first_grad is None or not visual_gradient_seen:
            gradients={key:sum(float(p.grad.float().norm()) for p in module.parameters() if p.grad is not None)
                        for key,module in [('a3',model.a3),('matcher',model.matcher),('trajectory_stop',model.trajectory)]}
            if first_grad is None:
                first_grad=gradients
                if first_grad['a3']<=0 or first_grad['trajectory_stop']<=0:raise ValueError('missing A3/navigation gradient')
                first_grad['visual_gain']=float(model.visual_gain.grad)
            visual_gradient_seen=visual_gradient_seen or gradients['matcher']>0
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step();scheduler.step()
        size=len(public['pose_world']);count+=size
        for key,value in loss.items():total[key]=total.get(key,0.)+float(value.detach())*size
        with torch.no_grad():
            distance=(grid_world-labels['target_world'][:,None,:2]).norm(dim=-1)
            ids=output.belief.belief_logits.topk(8,-1).indices
            ranked=distance.gather(1,ids)
            visible=output.observed
            match=output.appearance_scores.masked_fill(~visible,-1e4).argmax(-1)
            nearest=distance.argmin(-1)
            eligible=visible.gather(1,nearest[:,None]).squeeze(1)
            measures=dict(NE_m=float(ranked[:,0].sum()),
                **{f'R@{k}/20m_hits':float((ranked[:,:k].amin(-1)<=20).sum()) for k in (1,4,8)},
                appearance_eligible=float(eligible.sum()),appearance_identity_hits=float(((match==nearest)&eligible).sum()),
                observed_fraction=float(visible.float().sum()/visible.shape[-1]),
                stop_positive_states=float(((public['pose_world'][:,:2]-labels['target_world'][:,:2]).norm(dim=-1)<=20).sum()))
            for key,value in measures.items():total[key]=total.get(key,0.)+value
        if (index+1)%100==0:print(json.dumps(dict(batch=index+1,samples=count,seconds=time.perf_counter()-began)),flush=True)
        if index%100==0:
            gpu_utilization.append(float(subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu',
                '--format=csv,noheader,nounits'],text=True).strip().splitlines()[0]))
    torch.cuda.synchronize();seconds=time.perf_counter()-began
    averages={key:val/count for key,val in total.items() if key not in ('appearance_identity_hits','appearance_eligible')}
    averages['appearance_identity_accuracy']=total.get('appearance_identity_hits',0)/max(total.get('appearance_eligible',0),1)
    return dict(**averages,appearance_eligible_states=total.get('appearance_eligible',0),samples=count,
        seconds=seconds,samples_per_second=count/max(seconds,1e-9),first_batch_gradients=first_grad,
        visual_matching_gradient_observed=visual_gradient_seen,
        sampled_gpu_utilization_mean_percent=float(np.mean(gpu_utilization)),gpu_utilization_samples=len(gpu_utilization),
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('train-cache','a3-checkpoint','output','sbf-source','sbf-config','data-root'):p.add_argument('--'+key,required=True)
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--resume');p.add_argument('--max-batches',type=int)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--stop-after-epoch',type=int,help='Run full epochs with planned scheduler, then hold for review')
    a=p.parse_args();out=Path(a.output)
    if a.max_batches and not a.smoke:raise ValueError('partial epochs require explicit --smoke')
    if out.exists() and not a.resume:raise ValueError('refusing experiment overwrite')
    out.mkdir(parents=True,exist_ok=True);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    device=torch.device('cuda');data=JointActiveDataset(a.train_cache)
    if data.manifest['protocol']['split']!='train_seen':raise ValueError('train split only')
    if data.manifest['protocol']['a3_sha256']!=file_sha256(a.a3_checkpoint):raise ValueError('A3 source checkpoint mismatch')
    baseline=torch.load(a.a3_checkpoint,map_location='cpu',weights_only=True)
    from types import SimpleNamespace
    a3_args=baseline['args'];a3=new_model(SimpleNamespace(**a3_args),device);a3.load_state_dict(baseline['model'],strict=True)
    model=JointActiveGraphNav(a3).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
    steps=math.ceil(len(data)/a.batch_size)*a.epochs;warmup=max(1,int(.05*steps))
    def lr_factor(step):
        if step<warmup:return (step+1)/warmup
        return .5*(1+math.cos(math.pi*min(1,(step-warmup)/max(steps-warmup,1))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_factor)
    fingerprint=dict(train_cache=dict(path=str(Path(a.train_cache).resolve()),sha256=file_sha256(Path(a.train_cache)/'manifest.json')),
        a3_checkpoint=dict(path=str(Path(a.a3_checkpoint).resolve()),sha256=file_sha256(a.a3_checkpoint)),
        sbf_config=dict(path=str(Path(a.sbf_config).resolve()),sha256=file_sha256(a.sbf_config)))
    history=[];start=1;best=(-1.,-1.);source=provenance();pending=None
    if a.resume:
        state=restore_training(a.resume,model,optimizer,a,fingerprint,
            ignored=('resume','epochs','device','workers','stop_after_epoch'))
        scheduler.load_state_dict(state['scheduler']);history=state['history'];start=state['epoch']+1;best=tuple(state['best_seen']);source=state['provenance']
        if not state.get('epoch_validation_complete',True):
            pending=history[-1];history=history[:-1];start=state['epoch']
    bert=FrozenBERTTokens(name=a3_args['text_backbone'],max_length=a3_args.get('max_text_length',96))
    generator=torch.Generator()
    sampler=ShardShuffleSampler(data,seed=a.seed)
    loader=DataLoader(data,batch_size=a.batch_size,sampler=sampler,generator=generator,num_workers=a.workers,
        persistent_workers=a.workers>0,pin_memory=True,collate_fn=collate_joint)
    for epoch in range(start,a.epochs+1):
        generator.manual_seed(a.seed+epoch)
        sampler.epoch=epoch
        if pending is not None and pending['epoch']==epoch:
            record=pending;pending=None
        else:record=dict(epoch=epoch,train=run_epoch(model,loader,bert,device,optimizer,scheduler,max_batches=a.max_batches))
        protocol=dict(name='joint-active-v1',backbone='google/siglip2-large-patch16-384',
            backbone_revision=data.manifest['backbone_revision'],trainable_a3=True,freeze_bert=True,freeze_siglip=True,
            selected_by='Best Val Seen fixed128 SR then SPL among20 epochs; no Unseen selection',
            stop_positive_weight=20.,stop_threshold=.5,active_max_actions=2,horizon=20,
            horizontal_step_m=50.,vertical_step_m=10.,initialization='A3 historical 5-epoch checkpoint; extra warmstart compute disclosed',
            original_a3_sha256=file_sha256(a.a3_checkpoint))
        state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            epoch=epoch,args=vars(a),a3_args=a3_args,history=history+[record],best_seen=list(best),
            provenance=source,input_fingerprint=fingerprint,protocol=protocol,rng=rng_state())
        state['epoch_validation_complete']=False
        atomic_save(state,out/'last.pt')
        if not a.smoke:
            destination=out/f'seen128_epoch{epoch:02d}'
            command=[sys.executable,'scripts/evaluate_closedloop.py','--model','joint-active',
                '--sbf-source',a.sbf_source,'--config',a.sbf_config,'--checkpoint',str(out/'last.pt'),
                '--data-root',a.data_root,'--split','val_seen','--max-episodes','128','--output',str(destination)]
            started=time.perf_counter()
            with (out/f'seen128_epoch{epoch:02d}.log').open('w') as stream:
                subprocess.run(command,check=True,stdout=stream,stderr=subprocess.STDOUT)
            evaluation=json.loads((destination/'metrics.json').read_text())
            record['seen128']=evaluation['metrics'];record['seen128_goal_retrieval']=evaluation['goal_retrieval']
            record['seen128_active_action_counts']=evaluation['active_action_counts']
            record['validation_seconds']=time.perf_counter()-started
            key=(record['seen128']['SR'],record['seen128']['SPL'])
        else:key=(-record['train']['total'],0.)
        improved=key>best
        if improved:best=key
        history.append(record);state.update(history=history,best_seen=list(best),rng=rng_state(),epoch_validation_complete=True)
        atomic_save(state,out/'last.pt')
        if improved:atomic_save(state,out/'best.pt')
        (out/'history.json').write_text(json.dumps(history,indent=2))
        rows=[dict(epoch=h['epoch'],**{k:v for k,v in h['train'].items() if not isinstance(v,dict)},
                   **{'seen128_'+k:v for k,v in h.get('seen128',{}).items() if isinstance(v,(int,float))},
                   validation_seconds=h.get('validation_seconds')) for h in history]
        with (out/'training.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
            writer.writeheader();writer.writerows(rows)
        print(json.dumps(record),flush=True)
        if a.stop_after_epoch is not None and epoch>=a.stop_after_epoch:
            (out/'review_hold.json').write_text(json.dumps(dict(completed_epoch=epoch,planned_epochs=a.epochs,
                reason='User requests complete epoch1 metrics and timing before deciding subsequent epochs',
                last_checkpoint=str(out/'last.pt'),best_checkpoint=str(out/'best.pt')),indent=2))
            return
    (out/'training_complete.json').write_text(json.dumps(dict(epochs=a.epochs,smoke=a.smoke,
        train_samples=len(data),provenance=source,input_fingerprint=fingerprint,best_seen=list(best)),indent=2))


if __name__=='__main__':main()
