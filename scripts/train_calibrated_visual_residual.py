"""Train only an observed evidence residual; frozen V1 geometry/control, Train-only labels."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from graphnav.calibrated_visual_residual import CalibratedVisualResidual,correction_loss
from graphnav.stage1 import GraphNavStage1
from graphnav.stage1_data import Stage1Dataset,collate_stage1
from graphnav.full_experiment import new_model,_batch_inputs
from graphnav.token_language import FrozenBERTTokens
from graphnav.checkpointing import atomic_save,atomic_json,file_sha256
from graphnav.stage1_checkpoint import module_sha
from graphnav.workspace_paths import PROJECT,current_paths
from scripts.train_graphnav_stage1 import TrainingSampler


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='calibrated_residual');p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--rank-weight',type=float,default=0.,help='Top1 legal versus incorrect goal margin supervision')
    p.add_argument('--shuffle-regularization',action='store_true',help='Discourage geometry-only corrections and train real/shuffled image discrimination')
    p.add_argument('--native-contrast',action='store_true',help='Replace ego-distance shortcut with raw native RGB/text cosine evidence')
    p.add_argument('--uncentered',action='store_true',help='Permit absolute negative evidence on supported views, keep unseen cells unchanged')
    p.add_argument('--max-batches',type=int);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();out=Path(a.output)
    if a.max_batches and not a.smoke:raise ValueError('Partial training must be explicit smoke')
    if out.exists():raise ValueError('Refusing to overwrite residual training')
    out.mkdir(parents=True);torch.manual_seed(0);np.random.seed(0)
    checkpoint=PROJECT/'GraphNav/runs/graphnav_v1_stage1/best.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=True);device=torch.device('cuda')
    original=GraphNavStage1(new_model(SimpleNamespace(**state['a3_args']),device),residual_bound=.5).to(device)
    original.load_state_dict(state['model'],strict=True);original.requires_grad_(False).eval()
    model=CalibratedVisualResidual(centered=not a.uncentered,use_pose=not a.native_contrast).to(device)
    frozen_sha={name:module_sha(getattr(original,name)) for name in ('a3','matcher','history','trajectory')}
    cache=PROJECT/'GraphNav/data/graphnav_stage1_frames';data=Stage1Dataset(cache,verify=not a.smoke)
    dev=set(state['config']['dev_episodes']);episodes=data.manifest['protocol']['episode_indices']
    train_indices=TrainingSampler(data,0,dev);dev_indices=[i for i in range(len(data)) if episodes[i//4] in dev]
    common=dict(batch_size=a.batch_size,num_workers=a.workers,collate_fn=collate_stage1,pin_memory=True,persistent_workers=a.workers>0)
    train=DataLoader(data,sampler=train_indices,**common);validation=DataLoader(Subset(data,dev_indices),shuffle=False,**common)
    bert=FrozenBERTTokens(state['a3_args']['text_backbone'],max_length=96)
    bert.load_cache(current_paths(state['a3_args']['train_bert_cache']))
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01)
    config=dict(rank_weight=a.rank_weight,shuffle_regularization=a.shuffle_regularization,model_kwargs=dict(centered=not a.uncentered,use_pose=not a.native_contrast),base_checkpoint=str(checkpoint),base_sha=file_sha256(checkpoint),cache_manifest_sha=file_sha256(cache/'manifest.json'),epochs=a.epochs,train_states=len(train_indices),dev_states=len(dev_indices),dev_episodes=sorted(dev),selection='Train Dev belief R@1, then mass loss; no Unseen/Test selection',smoke=a.smoke,architecture='Frozen native current/history vectors x text interaction + frozen geometric context; learned quality/confidence; bounded 2.0 logit residual; 20m legal goal probability mass supervision')
    atomic_json(config,out/'config.json');history=[];best=(-1.,float('-inf'))
    def run(loader,training):
        model.train(training);sums={};count=0;started=time.perf_counter()
        for iteration,(static,public,labels) in enumerate(loader):
            if a.max_batches and iteration>=a.max_batches:break
            inputs,tokens,mask,names=_batch_inputs(static,bert,device,'geometry')
            public={k:v.to(device) for k,v in public.items()};labels={k:v.to(device) for k,v in labels.items()}
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                base=original(static_inputs=inputs,tokens=tokens,token_mask=mask,node_emb=names,**public,use_history=True,visual_control='no-belief')
            with torch.set_grad_enabled(training),torch.autocast('cuda',dtype=torch.bfloat16):
                delta,confidence,known=model(base,public,inputs)
            world=public['bounds'][:,None,:2]+inputs['grid_xy']*(public['bounds'][:,None,2:]-public['bounds'][:,None,:2])
            losses=correction_loss(base.static.belief_logits,delta,world,labels['target_world'],known,rank_weight=a.rank_weight)
            if training and a.shuffle_regularization:
                fake=dict(public)
                for feature_key,valid_key in (('roi_features','roi_valid'),('history_features','history_valid')):
                    values=public[feature_key].clone();valid=public[valid_key]
                    for b in range(len(values)):
                        for scale in range(3):
                            ids=valid[b,:,scale].nonzero().flatten()
                            permutation=torch.randperm(len(ids),device=ids.device)
                            values[b,ids,scale]=values[b,ids[permutation],scale]
                    fake[feature_key]=values
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    counterfactual=original(static_inputs=inputs,tokens=tokens,token_mask=mask,node_emb=names,**fake,use_history=True,visual_control='no-belief',base_field=base.static)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    fake_delta,_,fake_known=model(counterfactual,fake,inputs)
                legal=(world-labels['target_world'][:,None,:2]).norm(dim=-1)<=20
                eligible=(legal&known).any(-1)
                def nll(scores):
                    return torch.logsumexp(scores,-1)-torch.logsumexp(scores.masked_fill(~legal,-1e4),-1)
                contrast=torch.relu(.05+nll(base.static.belief_logits.float()+delta)-nll(base.static.belief_logits.float()+fake_delta))
                contrast=contrast[eligible].mean() if eligible.any() else delta.sum()*0
                fake_drift=(fake_delta.square()*fake_known).sum()/fake_known.sum().clamp_min(1)
                losses['total']=losses['total']+.1*contrast+.03*fake_drift
                losses['contrast']=contrast;losses['fake_drift']=fake_drift
            if not torch.isfinite(losses['total']):raise ValueError('Nonfinite correction loss')
            if training:
                optimizer.zero_grad(set_to_none=True);losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            with torch.no_grad():
                size=len(world);count+=size
                distance=(world-labels['target_world'][:,None,:2]).norm(dim=-1)
                for name,scores in [('static',base.static.belief_logits),('corrected',base.static.belief_logits.float()+delta)]:
                    winner=scores.float().argmax(-1);error=distance.gather(1,winner[:,None])[:,0]
                    sums[name+'_R1']=sums.get(name+'_R1',0.)+float((error<=20).sum())
                    sums[name+'_NE']=sums.get(name+'_NE',0.)+float(error.sum())
                for key,value in losses.items():sums[key]=sums.get(key,0.)+float(value.detach())*size
                sums['mean_abs_delta']=sums.get('mean_abs_delta',0.)+float(delta.abs().mean())*size
                sums['observed_fraction']=sums.get('observed_fraction',0.)+float(known.float().mean())*size
            if (iteration+1)%100==0:print(json.dumps(dict(training=training,batch=iteration+1,samples=count,elapsed=time.perf_counter()-started)),flush=True)
        return dict(**{k:v/count for k,v in sums.items()},samples=count,seconds=time.perf_counter()-started)
    for e in range(1,a.epochs+1):
        train_indices.base.epoch=e
        record=dict(epoch=e,train=run(train,True),dev=run(validation,False));history.append(record)
        key=(record['dev']['corrected_R1'],-record['dev']['mass'])
        payload=dict(kind='calibrated-visual-residual-v2',model=model.state_dict(),base_sha=config['base_sha'],config=config,history=history,epoch=e,optimizer=optimizer.state_dict())
        atomic_save(payload,out/'last.pt')
        if key>best:best=key;atomic_save(payload,out/'best.pt')
        atomic_json(history,out/'history.json');print(json.dumps(record),flush=True)
    assert frozen_sha=={name:module_sha(getattr(original,name)) for name in frozen_sha},'Frozen V1 changed'
    atomic_json(dict(complete=True,epochs=a.epochs,best_key=best,frozen_integrity=frozen_sha),out/'complete.json')


if __name__=='__main__':main()
