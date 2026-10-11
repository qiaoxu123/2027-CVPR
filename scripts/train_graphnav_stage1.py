"""Full A/B/C epochs with strict pretrained migration, causal inputs and Seen selection."""
import argparse
import copy
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
from torch.utils.data import DataLoader,Sampler,Subset
from graphnav.checkpointing import atomic_save,atomic_json,file_sha256,provenance,rng_state,restore_rng
from graphnav.full_experiment import _batch_inputs
from graphnav.joint_active_data import ShardShuffleSampler
from graphnav.stage1 import stage1_losses,visual_supervision_regions
from graphnav.stage1_checkpoint import initialize_model,module_sha,protocol,EXPECTED
from graphnav.stage1_data import Stage1Dataset,collate_stage1
from graphnav.stage1_runtime import retrieval_ids
from graphnav.token_language import FrozenBERTTokens


class TrainingSampler(Sampler):
    def __init__(self,data,seed,dev):self.data=data;self.base=ShardShuffleSampler(data,seed);self.dev=dev
    def __iter__(self):
        for index in self.base:
            if self.data.manifest['protocol']['episode_indices'][index//4] not in self.dev:yield index
    def __len__(self):return len(self.data)-4*len(self.dev)


def epoch(model,loader,bert,device,stage,optimizer=None,scheduler=None,max_batches=None,teacher=None):
    model.train(optimizer is not None);total={};count=0;began=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    visual_eligible=0;rank_gain=0.;appearance_hit=0.;gradient_norms={}
    for iteration,(static,public,labels)in enumerate(loader):
        if max_batches and iteration>=max_batches:break
        x,t,m,n=_batch_inputs(static,bert,device,'geometry');public={k:v.to(device,non_blocking=True)for k,v in public.items()};labels={k:v.to(device,non_blocking=True)for k,v in labels.items()}
        with torch.set_grad_enabled(optimizer is not None),torch.autocast('cuda',dtype=torch.bfloat16):
            output=model(static_inputs=x,tokens=t,token_mask=m,node_emb=n,**public,use_history=stage!='A')
        teacher_logits=None
        if teacher is not None:
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):teacher_logits=teacher(x,variant='geometry',tokens=t,token_mask=m,node_emb=n).belief_logits
        world=public['bounds'][:,None,:2]+x['grid_xy']*(public['bounds'][:,None,2:]-public['bounds'][:,None,:2])
        scales=public['roi_valid']|(public['history_valid']if stage!='A'else torch.zeros_like(public['roi_valid']))
        losses=stage1_losses(output,grid_world=world,pose_world=public['pose_world'],bounds=public['bounds'],**labels,observed_scales=scales,teacher_logits=teacher_logits,stage=stage)
        if not torch.isfinite(losses['total']):raise ValueError('nonfinite Stage1 loss')
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True);losses['total'].backward()
            if iteration==0:
                gradient_norms={k:sum(float(p.grad.float().norm())for p in module.parameters()if p.grad is not None)for k,module in [('a3',model.a3),('visual',model.matcher),('history',model.history),('trajectory_stop',model.trajectory)]}
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();scheduler.step()
        size=len(world);count+=size
        for key,value in losses.items():total[key]=total.get(key,0.)+float(value.detach())*size
        with torch.no_grad():
            distance=(world-labels['target_world'][:,None,:2]).norm(dim=-1)
            for name,field in [('static',output.static),('belief',output.belief)]:
                ids=retrieval_ids(field.belief_logits);dist=distance.gather(1,ids)
                total[name+'_NE_m']=total.get(name+'_NE_m',0.)+float(dist[:,0].sum())
                for k in (1,4,8,16):total[name+f'_R@{k}/20m']=total.get(name+f'_R@{k}/20m',0.)+float((dist[:,:k].amin(-1)<=20).sum())
            legal=distance<=20.;visible_legal,_=visual_supervision_regions(world,labels['target_world'],scales);eligible=visible_legal.any(-1)
            coverage=dict(current_observed_fraction=float(output.current_observed.float().mean()),
                belief_observed_fraction=float(output.observed.float().mean()),
                states_with_no_current_roi=float((~output.current_observed.any(-1)).float().mean()),
                history_observed_fraction=float(public['history_valid'].any(-1).float().mean()),
                residual_max_abs=float(output.delta.abs().max()))
            for key,value in coverage.items():total[key]=total.get(key,0.)+value*size
            visual_eligible+=int(eligible.sum())
            match=output.appearance_scores.masked_fill(~output.observed,-1e4).argmax(-1)
            appearance_hit+=float((legal.gather(1,match[:,None]).squeeze(1)&eligible).sum())
            ranks=[]
            for field in (output.static,output.belief):
                scores=field.belief_logits.float();best=scores.masked_fill(~legal,-1e4).amax(-1)
                ranks.append((scores>best[:,None]).sum(-1)+1)
            rank_gain+=float((ranks[0]-ranks[1])[eligible].sum())
        if (iteration+1)%100==0:print(json.dumps(dict(batch=iteration+1,samples=count,seconds=time.perf_counter()-began)),flush=True)
    torch.cuda.synchronize();seconds=time.perf_counter()-began
    return dict(**{k:v/count for k,v in total.items()},samples=count,seconds=seconds,samples_per_second=count/seconds,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,appearance_eligible_states=visual_eligible,
        appearance_correct_at20m=appearance_hit/max(visual_eligible,1),correct_candidate_rank_gain=rank_gain/max(visual_eligible,1),
        first_batch_gradient_norms=gradient_norms)


def main():
    p=argparse.ArgumentParser()
    for key in ('cache','sources','stage','output','sbf-source','sbf-config','data-root'):p.add_argument('--'+key,required=True)
    p.add_argument('--init');p.add_argument('--resume');p.add_argument('--epochs',type=int,required=True)
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--workers',type=int,default=4);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--max-batches',type=int);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();out=Path(a.output)
    if a.max_batches and not a.smoke:raise ValueError('partial epoch must be explicitly smoke')
    if out.exists() and not a.resume:raise ValueError('refusing stage overwrite')
    out.mkdir(parents=True,exist_ok=True);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    source=json.loads(Path(a.sources).read_text());paths={k:v['path']for k,v in source.items()};device=torch.device('cuda')
    model,a3_args=initialize_model(paths,device,a.init);model.configure_stage(a.stage)
    data=Stage1Dataset(a.cache)
    if data.manifest['smoke_only'] and not a.smoke:raise ValueError('limited features cannot be full training')
    if data.manifest['protocol']['a3_sha256']!=EXPECTED['a3']:raise ValueError('feature geometry was not validated for Epoch14')
    episodes=data.manifest['protocol']['episode_indices'][:len(data)//4]
    # Hold out whole map+instruction groups, including duplicated episodes.
    sys.path.insert(0,a.sbf_source)
    from sbfnav.dataset import CityNavDataset
    source_data=CityNavDataset(a.data_root,'train_seen',version='processed_citynav')
    if file_sha256(source_data.annotation_path)!=data.manifest['protocol']['source_sha256']:
        raise ValueError('Train Dev public query grouping source SHA mismatch')
    keys=sorted({(source_data[int(i)].map_name,source_data[int(i)].instruction)for i in episodes})
    rng=np.random.default_rng(a.seed)
    dev_keys={keys[int(i)]for i in rng.choice(len(keys),max(1,int(.05*len(keys))),replace=False)}
    dev={int(i)for i in episodes if (source_data[int(i)].map_name,source_data[int(i)].instruction)in dev_keys}
    dev_indices=[i for i in range(len(data))if episodes[i//4]in dev]
    sampler=TrainingSampler(data,a.seed,dev)
    generator=torch.Generator();common=dict(batch_size=a.batch_size,num_workers=a.workers,pin_memory=True,collate_fn=collate_stage1,persistent_workers=a.workers>0)
    train=DataLoader(data,sampler=sampler,generator=generator,**common);validation=DataLoader(Subset(data,dev_indices),shuffle=False,**common)
    groups=[dict(params=list(model.matcher.parameters())+[model.visual_gain],lr=1e-4,name='visual')]
    if a.stage!='A':groups.append(dict(params=model.history.parameters(),lr=1e-4,name='history'))
    if a.stage=='C':groups.extend([dict(params=[p for p in model.a3.parameters()if p.requires_grad],lr=1e-5,name='a3'),dict(params=model.trajectory.parameters(),lr=2e-5,name='trajectory_stop')])
    optimizer=torch.optim.AdamW(groups,weight_decay=.01)
    steps=math.ceil(len(sampler)/a.batch_size)*a.epochs;warmup=max(1,int(.05*steps))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:(s+1)/warmup if s<warmup else .5*(1+math.cos(math.pi*min(1,(s-warmup)/max(steps-warmup,1)))))
    fingerprint=dict(cache_manifest=file_sha256(Path(a.cache)/'manifest.json'),sources=source,sbf_config=file_sha256(a.sbf_config),init=file_sha256(a.init)if a.init else None)
    history=[];best=(-1e30,-1e30,-1e30);start=1;pending=None;resume_rng=None
    if a.init and not a.smoke:
        prior=torch.load(a.init,map_location='cpu',weights_only=True)
        if prior['args'].get('smoke'):raise ValueError('Smoke checkpoint cannot initialize full Stage1 training')
    frozen=dict(a3=module_sha(model.a3),trajectory=module_sha(model.trajectory))
    if a.resume:
        state=torch.load(a.resume,map_location='cpu',weights_only=True)
        if state['input_fingerprint']!=fingerprint:raise ValueError('resume input SHA differs')
        for k,v in vars(a).items():
            if k not in ('resume','workers')and state['args'].get(k)!=v:raise ValueError('resume configuration differs: '+k)
        model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);resume_rng=state['rng']
        history=state['history'];best=tuple(state['best_key']);start=state['epoch']+1
        if not state['validation_complete']:pending=history[-1];history=history[:-1];start=state['epoch']
    bert=FrozenBERTTokens(a3_args['text_backbone'],max_length=a3_args.get('max_text_length',96))
    baseline=torch.load(paths['a3'],map_location='cpu',weights_only=True)
    train_cache=baseline['input_fingerprint']['train_bert_cache']['path'];seen_cache=baseline['input_fingerprint']['seen_bert_cache']['path']
    bert.load_cache(train_cache);embeddings=bert.cache.copy();bert.load_cache(seen_cache);bert.cache.update(embeddings);del embeddings
    teacher=copy.deepcopy(model.a3).requires_grad_(False).eval()if a.stage=='C'else None
    if teacher is not None:teacher.load_state_dict(baseline['model'],strict=True)
    if resume_rng is not None:restore_rng(resume_rng)
    config=dict(args=vars(a),a3_args=a3_args,protocol=protocol(data.manifest['backbone_revision'],a.stage),sources=source,
        frozen_backbones=dict(bert_revision=bert.revision,siglip_revision=data.manifest['backbone_revision']),
        losses=dict(goal=1.,visual=.1,belief=1.,rank=.05,trajectory=.1,trajectory_mode=.02,trajectory_rank=.05,stop=.05,anchor=1.),
        train_states=len(sampler),dev_states=len(dev_indices),dev_episodes=sorted(dev),dev_public_queries=sorted(dev_keys),seed=a.seed,
        parameter_counts={k:sum(p.numel()for p in module.parameters())for k,module in [('a3',model.a3),('visual',model.matcher),('history',model.history),('trajectory_stop',model.trajectory)]},
        actual_trainable_parameters=sum(p.numel()for p in model.parameters()if p.requires_grad),frozen_bert_parameters=sum(p.numel()for p in bert.model.parameters()))
    import yaml
    (out/'config.yaml').write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False))
    for e in range(start,a.epochs+1):
        generator.manual_seed(a.seed+e);sampler.base.epoch=e
        reusing_pending=pending is not None
        if reusing_pending:record=pending;pending=None
        else:
            record=dict(epoch=e,stage=a.stage,train=epoch(model,train,bert,device,a.stage,optimizer,scheduler,a.max_batches,teacher),
                        dev=epoch(model,validation,bert,device,a.stage,max_batches=a.max_batches,teacher=teacher))
        if a.stage in ('A','B'):
            if module_sha(model.a3)!=frozen['a3']or module_sha(model.trajectory)!=frozen['trajectory']:raise ValueError('frozen strong baseline parameters changed')
        state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=rng_state(),epoch=e,
            args=vars(a),a3_args=a3_args,source_sha256=EXPECTED,protocol=config['protocol'],input_fingerprint=fingerprint,
            history=history+[record],best_key=list(best),validation_complete=False,provenance=provenance(),config=config)
        if not reusing_pending:atomic_save(state,out/'last.pt')
        if not a.smoke:
            destination=out/f'seen128_epoch{e:02d}'
            if not (destination/'metrics.json').exists():
                if destination.exists():destination.rename(out/f'seen128_epoch{e:02d}_failed_{int(time.time())}')
                command=[sys.executable,'-m','scripts.evaluate_closedloop','--model','graphnav-v1','--sbf-source',a.sbf_source,'--config',a.sbf_config,'--checkpoint',str(out/'last.pt'),'--data-root',a.data_root,'--split','val_seen','--max-episodes','128','--output',str(destination)]
                with (out/f'seen128_epoch{e:02d}.log').open('a')as stream:subprocess.run(command,check=True,stdout=stream,stderr=subprocess.STDOUT)
            evaluation=json.loads((destination/'metrics.json').read_text())
            if evaluation['checkpoint_sha256']!=file_sha256(out/'last.pt'):raise ValueError('validation checkpoint differs')
            record['seen128']=evaluation['metrics'];record['retrieval']=evaluation['goal_retrieval']
            record['seen128_checkpoint_sha256']=evaluation['checkpoint_sha256'];record['episode_manifest_sha256']=evaluation['episode_manifest_sha256']
            key=(record['seen128']['SR'],record['seen128']['SPL'],record['dev']['belief_R@1/20m'])
        else:key=(-record['dev']['total'],0.,0.)
        improved=key>best
        if improved:best=key
        history.append(record);state.update(history=history,best_key=list(best),validation_complete=True,rng=rng_state())
        atomic_save(state,out/'last.pt')
        if improved:atomic_save(state,out/'best.pt')
        atomic_json(history,out/'history.json')
        rows=[dict(epoch=h['epoch'],stage=h['stage'],**{split+'_'+k:v for split in ('train','dev','seen128')for k,v in h.get(split,{}).items()if isinstance(v,(float,int))})for h in history]
        with (out/'training.csv').open('w')as f:
            w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
        print(json.dumps(record),flush=True)
    atomic_json(dict(stage=a.stage,epochs=a.epochs,smoke=a.smoke,train_states=len(sampler),dev_states=len(dev_indices),best_key=list(best),frozen_integrity=frozen),out/'complete.json')


if __name__=='__main__':main()
