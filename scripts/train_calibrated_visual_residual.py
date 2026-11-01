"""Train only an observed evidence residual; frozen V1 geometry/control, Train-only labels."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time
import re
import hashlib
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from graphnav.calibrated_visual_residual import CalibratedVisualResidual,correction_loss,observed_hard_ranking
from graphnav.stage1 import GraphNavStage1
from graphnav.stage1_data import Stage1Dataset,collate_stage1
from graphnav.full_experiment import new_model,_batch_inputs
from graphnav.token_language import FrozenBERTTokens
from graphnav.checkpointing import atomic_save,atomic_json,file_sha256
from graphnav.stage1_checkpoint import module_sha
from graphnav.workspace_paths import PROJECT,current_paths
from scripts.train_graphnav_stage1 import TrainingSampler
from graphnav.relation_roles import role_batch,parse_roles,BalancedRoleSampler
from graphnav.relation_supervision import compass_counterfactual_loss


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='calibrated_residual');p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--rank-weight',type=float,default=0.,help='Top1 legal versus incorrect goal margin supervision')
    p.add_argument('--shuffle-regularization',action='store_true',help='Discourage geometry-only corrections and train real/shuffled image discrimination')
    p.add_argument('--native-contrast',action='store_true',help='Replace ego-distance shortcut with raw native RGB/text cosine evidence')
    p.add_argument('--uncentered',action='store_true',help='Permit absolute negative evidence on supported views, keep unseen cells unchanged')
    p.add_argument('--max-batches',type=int);p.add_argument('--smoke',action='store_true')
    p.add_argument('--max-train-batches',type=int,help='Smoke training budget; keep the full Dev for selection and controls')
    p.add_argument('--relation',action='store_true',help='Observed visual anchor relation messages')
    p.add_argument('--relation-attention',action='store_true',help='Language/geometry-conditioned attention with spatially diverse anchors')
    p.add_argument('--hard-rank-weight',type=float,default=0.)
    p.add_argument('--structured-relation',action='store_true')
    p.add_argument('--official-relation',action='store_true',help='Actual pretrained official AerialVG all-pairs module with learned feature adapters')
    p.add_argument('--report-roles',action='store_true')
    p.add_argument('--role-balanced',action='store_true')
    p.add_argument('--role-cf-weight',type=float,default=0.)
    p.add_argument('--audit-groups',type=int,default=0,help='Reserve new residual-head audit groups from non-Dev Train; original backbone saw Train')
    p.add_argument('--cache',type=Path,default=PROJECT/'GraphNav/data/rebuilt_stage1_20261025/frames')
    p.add_argument('--bert-cache',type=Path,default=PROJECT/'GraphNav/data/rebuilt_stage1_20261025/bert_tokens.pt')
    p.add_argument('--evaluate-checkpoint',type=Path,help='Evaluate a trained head with visual controls, without optimization')
    a=p.parse_args();out=Path(a.output)
    if (a.max_batches or a.max_train_batches) and not a.smoke:raise ValueError('Partial training must be explicit smoke')
    if out.exists():raise ValueError('Refusing to overwrite residual training')
    out.mkdir(parents=True);torch.manual_seed(0);np.random.seed(0)
    checkpoint=PROJECT/'GraphNav/runs/graphnav_v1_stage1/best.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=True);device=torch.device('cuda')
    original=GraphNavStage1(new_model(SimpleNamespace(**state['a3_args']),device),residual_bound=.5).to(device)
    original.load_state_dict(state['model'],strict=True);original.requires_grad_(False).eval()
    model=CalibratedVisualResidual(centered=not a.uncentered,use_pose=not a.native_contrast,relation=a.relation,relation_attention=a.relation_attention,structured_relation=a.structured_relation,official_relation=a.official_relation).to(device)
    if a.evaluate_checkpoint:
        evaluated=torch.load(a.evaluate_checkpoint,map_location='cpu',weights_only=True)
        if evaluated['base_sha']!=file_sha256(checkpoint):raise ValueError('Base checkpoint mismatch')
        model=CalibratedVisualResidual(**evaluated['config']['model_kwargs']).to(device)
        model.load_state_dict(evaluated['model'],strict=True)
        a.relation=model.relation is not None
        a.official_relation=model.official_relation is not None
        a.epochs=0
        atomic_save(evaluated,out/'best.pt')
    frozen_sha={name:module_sha(getattr(original,name)) for name in ('a3','matcher','history','trajectory')}
    official_sha=module_sha(model.official_relation.pairs) if model.official_relation is not None else None
    cache=a.cache;data=Stage1Dataset(cache,verify=not a.smoke)
    dev=set(state['config']['dev_episodes']);episodes=data.manifest['protocol']['episode_indices']
    audit=set();embargo=set()
    if a.audit_groups:
        sys.path.insert(0,str(PROJECT/'Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source'))
        from sbfnav.dataset import CityNavDataset
        source=CityNavDataset(PROJECT.parent/'DATA/sbfnav-data','train_seen',version='processed_citynav')
        if file_sha256(source.annotation_path)!=data.manifest['protocol']['source_sha256']:raise ValueError('Audit source differs from cached source')
        keys={e:(source[e].map_name,source[e].instruction) for e in episodes}
        dev_keys={keys[e] for e in dev}
        embargo={e for e,key in keys.items() if key in dev_keys}
        candidates=set(keys.values())-dev_keys
        chosen=set(sorted(candidates,key=lambda key:hashlib.sha256(json.dumps(key).encode()).hexdigest())[:a.audit_groups])
        audit={e for e,key in keys.items() if key in chosen}
        atomic_json(dict(audit_episodes=sorted(audit),audit_groups=sorted(chosen),note='Held out from this new residual-head training/selection. Original backbone and past experiments used Train; not independent system generalization.'),out/'audit_split.json')
    train_indices=TrainingSampler(data,0,dev|audit|embargo);dev_indices=[i for i in range(len(data)) if episodes[i//4] in dev]
    if a.role_balanced:
        if not a.audit_groups:raise ValueError('Role-balanced sampling requires grouped metadata via --audit-groups')
        role_keys={(q['map_name'],q['instruction']) for q in data.static.records if parse_roles(q['instruction'],[x['name'] for x in q['landmarks']])['valid']}
        role_episodes={e for e,key in keys.items() if key in role_keys}
        train_indices=BalancedRoleSampler(train_indices,episodes,role_episodes)
    audit_indices=[i for i in range(len(data)) if episodes[i//4] in audit]
    common=dict(batch_size=a.batch_size,num_workers=a.workers,collate_fn=collate_stage1,pin_memory=True,persistent_workers=a.workers>0)
    train=DataLoader(data,sampler=train_indices,**common);validation=DataLoader(Subset(data,dev_indices),shuffle=False,**common)
    audit_loader=DataLoader(Subset(data,audit_indices),shuffle=False,**common) if audit_indices else None
    bert=FrozenBERTTokens(state['a3_args']['text_backbone'],max_length=96)
    bert.load_cache(a.bert_cache)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01)
    config=dict(rank_weight=a.rank_weight,shuffle_regularization=a.shuffle_regularization,model_kwargs=dict(centered=not a.uncentered,use_pose=not a.native_contrast,relation=a.relation),base_checkpoint=str(checkpoint),base_sha=file_sha256(checkpoint),cache_manifest_sha=file_sha256(cache/'manifest.json'),epochs=a.epochs,train_states=len(train_indices),dev_states=len(dev_indices),dev_episodes=sorted(dev),selection='Train Dev belief R@1, then mass loss; no Unseen/Test selection',smoke=a.smoke,architecture='Frozen native current/history vectors x text interaction + frozen geometric context; learned quality/confidence; bounded 2.0 logit residual; 20m legal goal probability mass supervision')
    config.update(cache_path=str(cache.resolve()),bert_cache_path=str(a.bert_cache.resolve()),bert_cache_sha=file_sha256(a.bert_cache),evaluate_checkpoint=str(a.evaluate_checkpoint) if a.evaluate_checkpoint else None)
    if a.evaluate_checkpoint:config['model_kwargs']=evaluated['config']['model_kwargs']
    else:config['model_kwargs']['relation_attention']=a.relation_attention
    if not a.evaluate_checkpoint:config['model_kwargs']['structured_relation']=a.structured_relation
    if not a.evaluate_checkpoint:config['model_kwargs']['official_relation']=a.official_relation
    config['role_cf_weight']=a.role_cf_weight;config['audit_episodes']=sorted(audit)
    config['role_balanced']=a.role_balanced;config['group_embargo_episodes']=sorted(embargo)
    if official_sha:
        config['official_pairs_sha']=official_sha
        config['official_source_sha']=file_sha256(PROJECT.parent/'third_party/AerialVG/model/AerialVG/relation.py')
        config['official_checkpoint_sha']=file_sha256(PROJECT.parent/'third_party/aerialvg_assets/aerialvg.pth')
    config['hard_rank_weight']=a.hard_rank_weight
    config['max_train_batches']=a.max_train_batches;config['max_batches']=a.max_batches
    atomic_json(config,out/'config.json');history=[];best=(-1.,float('-inf'))
    def run(loader,training,control='real',partition='dev'):
        model.train(training);sums={};count=0;started=time.perf_counter();predictions=[];strata={}
        for iteration,(static,public,labels) in enumerate(loader):
            if a.max_batches and iteration>=a.max_batches:break
            if training and a.max_train_batches and iteration>=a.max_train_batches:break
            inputs,tokens,mask,names=_batch_inputs(static,bert,device,'geometry')
            public={k:v.to(device) for k,v in public.items()};labels={k:v.to(device) for k,v in labels.items()}
            if model.structured_relation or a.report_roles:public.update(role_batch(static['text'],static['node_names'],bert,device)[0])
            real_support=torch.zeros(public['roi_valid'].shape[:2],device=device,dtype=torch.bool)
            for prefix in ('roi','history'):
                real_support|=(public[prefix+'_valid']&(public[prefix+'_features'].abs().sum(-1)>0)&(public[prefix+'_quality'][...,0]>0)&(public[prefix+'_quality'][...,1]>0)).any(-1)
            if control!='real':
                generator=torch.Generator(device=device).manual_seed(1000+iteration)
                for feature_key,valid_key in (('roi_features','roi_valid'),('history_features','history_valid')):
                    values=public[feature_key].clone()
                    if control=='zero':values.zero_()
                    elif control=='shuffle':
                        for b in range(len(values)):
                            for scale in range(values.shape[2]):
                                ids=public[valid_key][b,:,scale].nonzero().flatten()
                                order=torch.randperm(len(ids),device=device,generator=generator)
                                values[b,ids,scale]=values[b,ids[order],scale]
                    else:raise ValueError(control)
                    public[feature_key]=values
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                base=original(static_inputs=inputs,tokens=tokens,token_mask=mask,node_emb=names,**{k:v for k,v in public.items() if not k.startswith('role_')},use_history=True,visual_control='no-belief')
            with torch.set_grad_enabled(training),torch.autocast('cuda',dtype=torch.bfloat16):
                delta,confidence,known=model(base,public,inputs,tokens=tokens,token_mask=mask,node_emb=names)
            world=public['bounds'][:,None,:2]+inputs['grid_xy']*(public['bounds'][:,None,2:]-public['bounds'][:,None,:2])
            losses=correction_loss(base.static.belief_logits,delta,world,labels['target_world'],known,rank_weight=a.rank_weight)
            if training and a.role_cf_weight:
                cf=dict(public);cf['role_direction']=-public['role_direction']
                # Compass-only semantic negation; other relation kinds are masked by the loss.
                words=[{'north':'south','south':'north','east':'west','west':'east'}.get(r['relation'],r['relation']) for r in role_batch(static['text'],static['node_names'],bert,device)[1]]
                t,m=bert.encode_tokens(words);cf['role_relation']=((t*m[...,None]).sum(1)/m.sum(1,keepdim=True).clamp_min(1)).clone()
                with torch.autocast('cuda',dtype=torch.bfloat16):opposite=model(base,cf,inputs,tokens=tokens,token_mask=mask,node_emb=names)[0]
                auxiliary,eligible=compass_counterfactual_loss(delta,opposite,world,labels['target_world'],known,inputs['pair_geometry'],public)
                losses['role_cf']=auxiliary;losses['role_cf_eligible']=eligible.float().mean();losses['total']=losses['total']+a.role_cf_weight*auxiliary
            visual=(public['roi_features'].float()*public['roi_valid'][...,None]).sum(-2)+(public['history_features'].float()*public['history_valid'][...,None]).sum(-2)
            if a.hard_rank_weight:
                hard=observed_hard_ranking(base.static.belief_logits,delta,world,labels['target_world'],known,visual)
                losses['hard_rank']=hard;losses['total']=losses['total']+a.hard_rank_weight*hard
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
                    counterfactual=original(static_inputs=inputs,tokens=tokens,token_mask=mask,node_emb=names,**{k:v for k,v in fake.items() if not k.startswith('role_')},use_history=True,visual_control='no-belief',base_field=base.static)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    fake_delta,_,fake_known=model(counterfactual,fake,inputs,tokens=tokens,token_mask=mask,node_emb=names)
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
                legal=distance<=20
                observed=(legal&real_support).any(-1)
                positive=base.static.belief_logits.float().masked_fill(~(legal&real_support),-1e9).amax(-1)
                negative=base.static.belief_logits.float().masked_fill(~((~legal)&real_support),-1e9).amax(-1)
                relation_words=torch.tensor([bool(re.search(r'\b(north|south|east|west|left|right|between|opposite|beside|behind|near|adjacent)\b',text,re.I)) for text in static['text']],device=device)
                groups={'observed_target':observed,'observed_ambiguous':observed&((negative-positive).abs()<=1.),'relation_word_observed':relation_words&observed}
                if 'role_valid' in public:
                    reference_distance=inputs['pair_geometry'].float()[...,5].abs().gather(2,public['role_reference_index'].clamp_min(0)[:,None,None].expand(-1,world.shape[1],1)).squeeze(-1)
                    reference_region_observed=((reference_distance<.05)&real_support).any(-1)
                    groups['bound_role_observed']=public['role_valid']&observed
                    groups['target_reference_regions_observed']=public['role_valid']&observed&reference_region_observed
                for name,scores in [('static',base.static.belief_logits),('corrected',base.static.belief_logits.float()+delta)]:
                    winner=scores.float().argmax(-1);error=distance.gather(1,winner[:,None])[:,0]
                    sums[name+'_R1']=sums.get(name+'_R1',0.)+float((error<=20).sum())
                    sums[name+'_NE']=sums.get(name+'_NE',0.)+float(error.sum())
                    for group,selected in groups.items():
                        key=name+'/'+group
                        entry=strata.setdefault(key,dict(samples=0,hits=0,error_sum=0.))
                        entry['samples']+=int(selected.sum());entry['hits']+=int(((error<=20)&selected).sum());entry['error_sum']+=float(error[selected].sum())
                    if not training and name=='corrected':predictions.extend(error.cpu().tolist())
                for key,value in losses.items():sums[key]=sums.get(key,0.)+float(value.detach())*size
                sums['mean_abs_delta']=sums.get('mean_abs_delta',0.)+float(delta.abs().mean())*size
                sums['observed_fraction']=sums.get('observed_fraction',0.)+float(known.float().mean())*size
            if (iteration+1)%100==0:print(json.dumps(dict(training=training,batch=iteration+1,samples=count,elapsed=time.perf_counter()-started)),flush=True)
        if not count:raise ValueError('Empty training/evaluation loader')
        if not training:
            tag='no_relation' if (a.relation or a.official_relation) and model.relation is None and model.official_relation is None else control
            state_indices=dev_indices if partition=='dev' else audit_indices
            atomic_json(dict(dev_indices=state_indices[:count],errors_m=predictions,strata=strata),out/('predictions_'+('audit_' if partition=='audit' else '')+tag+'.json'))
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
    if official_sha:assert official_sha==module_sha(model.official_relation.pairs),'Frozen official AerialVG pair weights changed'
    model.load_state_dict(torch.load(out/'best.pt',map_location=device,weights_only=True)['model'])
    controls={name:run(validation,False,name) for name in ('real','zero','shuffle')}
    if a.relation or a.official_relation:
        relation=model.relation;official=model.official_relation;model.relation=None;model.official_relation=None
        controls['no_relation']=run(validation,False)
        model.relation=relation;model.official_relation=official
    atomic_json(controls,out/'controls.json')
    if audit_loader is not None:
        audit_controls={name:run(audit_loader,False,name,partition='audit') for name in ('real','zero','shuffle')}
        if a.relation or a.official_relation:
            relation=model.relation;official=model.official_relation;model.relation=None;model.official_relation=None
            audit_controls['no_relation']=run(audit_loader,False,partition='audit');model.relation=relation;model.official_relation=official
        atomic_json(audit_controls,out/'audit_controls.json')
    atomic_json(dict(complete=True,epochs=a.epochs,best_key=best,frozen_integrity=frozen_sha),out/'complete.json')


if __name__=='__main__':main()
