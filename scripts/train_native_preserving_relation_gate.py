"""Fixed-budget TRAIN-only relation/Gate v2; internal Dev only after freeze."""
import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from graphnav.checkpointing import atomic_json,atomic_save,file_sha256
from graphnav.gated_belief_fusion import ReliabilityGatedBelief,count_outcomes
from graphnav.native_multireference import (MultiReferenceRanker,full_public_graph,
    multireference_descriptors,auxiliary_reference_loss,belief_training_loss)
from graphnav.native_preserving_gate import (NativePreservingGate,incremental_targets,
    fit_probability_calibration,select_safe_threshold)
from graphnav.native_relation_grounding import weak_relation_ranking_loss
from graphnav.object_query_belief import ObjectQueryRanker
from graphnav.stage1_data import Stage1Dataset
from scripts.train_native_gated_fusion import split_train_by_group,tensor_batch
from scripts.train_native_relational_grounding import old_native
from scripts.train_object_query_belief import collate,load

REFERENCE_KEYS=('reference_features','reference_types','reference_valid','reference_binding')


def prepare(rows,static):
    lookup={}
    for i,record in enumerate(static.records):
        lookup.setdefault((record['map_name'],record['instruction']),i)
    cache={};output=[]
    for row in rows:
        key=(row['map_name'],row['instruction'])
        if key not in cache:
            index=lookup[key];graph=full_public_graph(static,index)
            cache[key]=graph
        graph=cache[key]
        public=multireference_descriptors(row['query_world'].float().numpy(),graph)
        output.append(dict(row,**public,public_entity_count=len(graph['entities']),
                           parsed_relations=len(graph['relations']),
                           ambiguous_relations=sum(any(len(a)!=1 for a in r.anchors) for r in graph['relations'])))
    return output


def split_probability_threshold(rows):
    """Group split stratified by PUBLIC relation availability, never GT/outcomes.

    Pure hashing placed all three named calibration groups on one side. Each
    side must have actual eligible relations before any thresholds can be fit.
    """
    grouped={}
    for row in rows:grouped.setdefault(row['group'],[]).append(row)
    sides=[[],[]]
    for named in [False,True]:
        groups=[group for group,items in grouped.items()
                if any(bool(r['reference_binding'].any()) for r in items)==named]
        groups.sort(key=lambda group:hashlib.sha256(str(group).encode()).digest())
        for i,group in enumerate(groups):sides[i%2].extend(grouped[group])
    if {r['group'] for r in sides[0]} & {r['group'] for r in sides[1]}:
        raise ValueError('Probability/threshold group overlap')
    if not all(any(r['reference_binding'].any() for r in side) for side in sides):
        raise ValueError('Insufficient named groups for independent calibration')
    return sides


def pack(rows):
    result=collate(rows)
    result.update({key:torch.stack([r[key] for r in rows]) for key in REFERENCE_KEYS})
    return result


def summary(rows):
    named=[r for r in rows if r['reference_binding'].any()]
    covered=[r for r in named if (((r['query_world'].float()-r['labels']['target_world'][:2]).norm(dim=-1)<=20)&r['valid']).any()]
    true_multi=[r for r in named if r['reference_binding'].sum()>1]
    distractor=[r for r in named if (r['reference_valid'] & ~r['reference_binding']).any()]
    multi=[r for r in named if r['reference_valid'].sum()>1]
    return dict(states=len(rows),groups=len({r['group'] for r in rows}),
                parsed_states=sum(r['parsed_relations']>0 for r in rows),
                ambiguous_only_states=sum(r['ambiguous_relations']>0 and not r['reference_binding'].any() for r in rows),
                named_states=len(named),named_groups=len({r['group'] for r in named}),
                covered_named_states=len(covered),covered_named_groups=len({r['group'] for r in covered}),
                true_multi_anchor_states=len(true_multi),true_multi_anchor_groups=len({r['group'] for r in true_multi}),
                with_distractor_states=len(distractor),multislot_states=len(multi),
                entity_count_min=min(r['public_entity_count'] for r in rows),
                entity_count_max=max(r['public_entity_count'] for r in rows))


def predict(model,x,q,mode='real'):
    features=x['reference_features']
    if mode=='shuffle':
        features=features.clone()
        for i in range(len(features)):
            ids=x['valid'][i].nonzero().flatten()
            features[i,ids]=features[i,ids.flip(0)]
    return model(x['features'],x['text'],x['text_mask'],x['geometry'],x['valid'],
                 features,x['reference_types'],x['reference_valid'],q,enabled=mode!='off')


def residual(native,x,q,null):
    return native.update_belief(x['base_logits'],q,null,x['kernel'],x['observed'],x['valid'],bound=2.)[0]


def gated(gate,x,native_logits,pred,null,calibration=None,threshold=1.,enabled=False):
    calibration=calibration or dict(slope=1.,bias=0.)
    return gate(native_logits,pred['logits'],null,x['kernel'],x['observed'],x['valid'],
                x['geometry'],x['grid_world'],pred['has_relation'],enabled=enabled,
                threshold=threshold,slope=calibration['slope'],bias=calibration['bias'])


def loader(rows,seed=None):
    return DataLoader(rows,batch_size=16,shuffle=seed is not None,collate_fn=pack,
                      generator=torch.Generator().manual_seed(seed or 0))


@torch.no_grad()
def calibration_data(rows,native,relation,gate,device,sigma,calibration=None):
    result={k:[] for k in ['raw','target','eligible','native_error','proposal_error','a0_error','probability']}
    for raw in loader(rows):
        x=tensor_batch(raw,device,sigma);q,null=old_native(native,x,device)
        n=residual(native,x,q,null);pred=predict(relation,x,q)
        output=gated(gate,x,n,pred,null,calibration)
        labels=incremental_targets(n,output['proposal_logits'],x['grid_world'],x['target_world'],output['eligible'])
        distance=(x['grid_world']-x['target_world'][:,None,:2]).norm(dim=-1)
        values=dict(raw=output['raw'],target=labels['target'],eligible=output['eligible'],
                    native_error=labels['before_error'],proposal_error=labels['proposal_error'],
                    a0_error=distance.gather(1,x['base_logits'].argmax(-1)[:,None]).squeeze(-1),
                    probability=output['probability'])
        for key,value in values.items():result[key].append(value.cpu())
    return {k:torch.cat(v) for k,v in result.items()}


@torch.no_grad()
def evaluate(rows,native,relation,gate,old_gate,device,sigma,calibration,threshold,old_threshold):
    totals={};ranks={};accepts={};details=[];attention_good=attention_count=0;binding_mass=0.;multi_count=multi_good=0;multi_mass=0.
    for raw in loader(rows):
        x=tensor_batch(raw,device,sigma);q,null=old_native(native,x,device)
        n=residual(native,x,q,null);variants={'A0':x['base_logits'],'native':n}
        for mode in ['real','off','shuffle']:
            pred=predict(relation,x,q,mode)
            direct=residual(native,x,pred['logits'],null)
            old=old_gate(x['base_logits'],pred['logits'],null,x['kernel'],x['observed'],x['valid'],
                         x['geometry'],x['grid_world'],enabled=True,threshold=old_threshold,transfer=.9)
            new=gated(gate,x,n,pred,null,calibration,threshold,True)
            for suffix,value in [('residual',direct),('old_gate',old['logits']),('new_gate',new['logits'])]:
                variants[mode+'_'+suffix]=value
            for label,value in [('old',old['accepted']),('new',new['accepted'])]:
                accepts[mode+'_'+label]=accepts.get(mode+'_'+label,0)+int(value.sum())
            if mode=='off':
                assert torch.equal(direct,n) and torch.equal(new['logits'],n), 'Native fallback violated'
            distance=(x['query_world']-x['target_world'][:,None,:2]).norm(dim=-1)
            correct=(distance.gather(1,pred['logits'].argmax(-1)[:,None]).squeeze(-1)<=20)&x['valid'].any(-1)
            ranks[mode]=ranks.get(mode,0)+int(correct.sum())
            if mode=='real':
                mask=x['valid'] & pred['has_relation'][:,None]
                selected=pred['attention'].argmax(-1)
                binding=x['reference_binding'][:,None].expand(-1,mask.shape[1],-1)
                attention_good+=int((binding.gather(-1,selected[:,:,None]).squeeze(-1)&mask).sum())
                attention_count+=int(mask.sum())
                binding_mass+=float((pred['attention']*binding).sum(-1)[mask].sum())
                multi_mask=mask & (x['reference_valid'].sum(-1)>1)[:,None]
                multi_count+=int(multi_mask.sum())
                multi_good+=int((binding.gather(-1,selected[:,:,None]).squeeze(-1)&multi_mask).sum())
                multi_mass+=float((pred['attention']*binding).sum(-1)[multi_mask].sum())
                for i in range(len(q)):
                    details.append(dict(group=raw['group'][i],episode=raw['episode'][i],step=raw['step'][i],
                        named=bool(pred['has_relation'][i]),reference_count=int(x['reference_valid'][i].sum()),
                        true_anchor_count=int(x['reference_binding'][i].sum()),accepted=bool(new['accepted'][i]),
                        gate_probability=float(new['probability'][i])))
        for name,value in variants.items():
            acc=totals.setdefault(name,{})
            for baseline,b in [('A0',x['base_logits']),('Native',n)]:
                for key,v in count_outcomes(b,value,x['grid_world'],x['target_world']).items():
                    acc[baseline+'_'+key]=acc.get(baseline+'_'+key,0)+v
    result={}
    for name,acc in totals.items():
        count=acc['A0_count'];correct=acc['A0_updated_correct']
        row=dict(R1=correct/count,correct=correct,NE=acc['A0_updated_ne_sum']/count)
        for baseline in ['A0','Native']:
            row[baseline+'_corrected']=acc[baseline+'_corrected'];row[baseline+'_damaged']=acc[baseline+'_damaged']
            row[baseline+'_net']=acc[baseline+'_corrected']-acc[baseline+'_damaged']
        mode=name.split('_')[0]
        row['candidate_correct']=None if name=='A0' else ranks.get('off' if name=='native' else mode)
        row['gate_accepted']=accepts.get(mode+'_'+('new' if name.endswith('new_gate') else 'old'),0) if 'gate' in name else 0
        row['gate_acceptance_rate']=row['gate_accepted']/len(rows)
        result[name]=row
    result['attention']=dict(query_count=attention_count,binding_top1_correct=attention_good,
                             binding_top1_accuracy=attention_good/max(attention_count,1),
                             mean_binding_mass=binding_mass/max(attention_count,1),
                             multislot_query_count=multi_count,multislot_top1_accuracy=multi_good/max(multi_count,1),
                             multislot_mean_binding_mass=multi_mass/max(multi_count,1))
    result['data_summary']=summary(rows)
    return result,details


def checked_metadata(directory,partition):
    path=Path(directory)/'manifest.json';meta=json.loads(path.read_text())
    if not meta.get('complete') or meta.get('partition')!=partition:
        raise ValueError('Only completed TRAIN/internal Dev cache allowed')
    return meta


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['train','dev','stage1-cache','native-checkpoint','old-pilot','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--epochs',type=int,default=4);p.add_argument('--gate-epochs',type=int,default=3)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--evaluate-dev',action='store_true')
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise ValueError('Refuse overwrite')
    if not 1<=a.epochs<=4 or not 1<=a.gate_epochs<=3:raise ValueError('Fixed pilot budget exceeded')
    train_meta=checked_metadata(a.train,'train');dev_meta=checked_metadata(a.dev,'dev')
    if train_meta['base_sha']!=dev_meta['base_sha']:raise ValueError('Provenance mismatch')
    source_sha=file_sha256(a.native_checkpoint);old_sha=file_sha256(a.old_pilot)
    frozen=torch.load(a.native_checkpoint,map_location='cpu',weights_only=True)
    old=torch.load(a.old_pilot,map_location='cpu',weights_only=True)
    if frozen['config']['base_sha']!=train_meta['base_sha'] or old['protocol']['native_checkpoint_sha']!=source_sha:raise ValueError('Frozen model mismatch')
    public_sha=file_sha256(Path(a.stage1_cache)/'queries.jsonl')
    _,train=load(a.train);static=Stage1Dataset(a.stage1_cache,verify=False).static
    train=prepare(train,static);fit,cal=split_train_by_group(train)
    probability_rows,threshold_rows=split_probability_threshold(cal)
    if not probability_rows or not threshold_rows:raise ValueError('Empty disjoint calibration split')
    protocol=dict(base_commit='b83287e17e34542ae4b1a8d407e1859149987c2e',seed=a.seed,epochs=a.epochs,gate_epochs=a.gate_epochs,
                  native_sha=source_sha,old_gate_sha=old_sha,public_sha=public_sha,
                  train_manifest_sha=file_sha256(Path(a.train)/'manifest.json'),dev_manifest_sha=file_sha256(Path(a.dev)/'manifest.json'),
                  source_sha={f:file_sha256(f) for f in [__file__,'graphnav/native_preserving_gate.py','graphnav/native_multireference.py']},
                  fit=summary(fit),probability_calibration=summary(probability_rows),threshold_selection=summary(threshold_rows),
                  policy='TRAIN fit + disjoint TRAIN probability/threshold groups; Dev once after freeze; no formal evaluation',
                  weights=dict(ranking=1.,attention=.2,geometry=.1,localization=.25,KL=.5,native_correct_margin=1.),
                  gate_target='R1 then >=1m NE gain vs Native; only unique named-relation eligible',
                  threshold_damage_budget=0,admitted_for_runtime=False)
    out.mkdir(parents=True);atomic_json(protocol,out/'protocol.json')
    torch.manual_seed(a.seed);device=torch.device('cuda');sigma=float(frozen['config']['kernel_sigma'])
    native=ObjectQueryRanker(relation=False).to(device);native.load_state_dict(frozen['model']);native.eval().requires_grad_(False)
    old_gate=ReliabilityGatedBelief().to(device);old_gate.load_state_dict(old['gate_state']);old_gate.eval().requires_grad_(False)
    relation=MultiReferenceRanker().to(device);optimizer=torch.optim.AdamW(relation.parameters(),lr=2e-4,weight_decay=.01)
    initial={k:v.detach().clone() for k,v in relation.named_parameters()};gradients={};history=[]
    for epoch in range(a.epochs):
        relation.train();sums={};steps=supervised=0
        for raw in loader(fit,a.seed+epoch):
            x=tensor_batch(raw,device,sigma);q,null=old_native(native,x,device)
            pred=predict(relation,x,q);n=residual(native,x,q,null).detach()
            updated=residual(native,x,pred['logits'],null)
            rank,stats=weak_relation_ranking_loss(pred['logits'],x['valid'],pred['has_relation'],x['query_world'],x['target_world'])
            attn,geo=auxiliary_reference_loss(pred,x)
            loc,kl,safe=belief_training_loss(updated,n,x,pred['has_relation'])
            losses=dict(ranking=rank,attention=attn,geometry=geo,localization=loc,KL=kl,native_correct_margin=safe)
            loss=sum(protocol['weights'][k]*v for k,v in losses.items())
            optimizer.zero_grad(set_to_none=True)
            if pred['has_relation'].any():
                if not torch.isfinite(loss):raise ValueError('Nonfinite relation loss')
                loss.backward()
                for name,parameter in relation.named_parameters():
                    if parameter.grad is not None:
                        key=name.split('.')[0];gradients[key]=max(gradients.get(key,0.),float(parameter.grad.norm()))
                torch.nn.utils.clip_grad_norm_(relation.parameters(),2.,error_if_nonfinite=True);optimizer.step();steps+=1
            supervised+=stats['supervised_rows']
            for k,v in losses.items():sums[k]=sums.get(k,0.)+float(v.detach())*len(q)
        record=dict(epoch=epoch+1,ranking_supervised_rows=supervised,optimizer_steps=steps,
                    **{k:v/len(fit) for k,v in sums.items()})
        history.append(record);atomic_json(history,out/'relation_history.json');print(json.dumps(record),flush=True)
    relation.eval().requires_grad_(False)
    atomic_json(dict(gradient_max=gradients,parameter_delta={k:float((v-initial[k]).norm()) for k,v in relation.named_parameters()},
                     frozen_native_has_gradient=any(v.grad is not None for v in native.parameters())),out/'gradient_audit.json')
    gate=NativePreservingGate().to(device)
    # Fit-label prior, not a Dev-derived initialization.
    fit_labels=calibration_data(fit,native,relation,gate,device,sigma)
    active=fit_labels['eligible'];positive=float(fit_labels['target'][active].sum());num=int(active.sum())
    prior=(positive+1)/(num+2)
    with torch.no_grad():gate.net[-1].bias.fill_(math.log(prior/(1-prior)))
    gate_optimizer=torch.optim.AdamW(gate.parameters(),lr=5e-4);gate_history=[]
    for epoch in range(a.gate_epochs):
        count=positive_count=steps=0;total=0.
        for raw in loader(fit,a.seed+100+epoch):
            x=tensor_batch(raw,device,sigma);q,null=old_native(native,x,device)
            with torch.no_grad():n=residual(native,x,q,null);pred=predict(relation,x,q)
            result=gated(gate,x,n,pred,null)
            labels=incremental_targets(n,result['proposal_logits'],x['grid_world'],x['target_world'],result['eligible'])
            mask=result['eligible'];gate_optimizer.zero_grad(set_to_none=True)
            if mask.any():
                loss=F.binary_cross_entropy_with_logits(result['raw'][mask],labels['target'][mask])
                loss.backward();torch.nn.utils.clip_grad_norm_(gate.parameters(),2.,error_if_nonfinite=True);gate_optimizer.step()
                total+=float(loss.detach())*int(mask.sum());count+=int(mask.sum());positive_count+=int(labels['target'][mask].sum());steps+=1
        record=dict(epoch=epoch+1,loss=total/max(count,1),eligible=count,positive=positive_count,steps=steps)
        gate_history.append(record);atomic_json(gate_history,out/'gate_history.json');print(json.dumps(record),flush=True)
    gate.eval().requires_grad_(False)
    probability=calibration_data(probability_rows,native,relation,gate,device,sigma)
    calibration=fit_probability_calibration(probability['raw'][probability['eligible']],probability['target'][probability['eligible']])
    selection=calibration_data(threshold_rows,native,relation,gate,device,sigma,calibration)
    selected=select_safe_threshold(selection['probability'],selection['eligible'],selection['native_error'],selection['proposal_error'],selection['a0_error'])
    if not calibration['class_complete']:
        selected['unvalidated_selection']=selected['selected']
        selected['selected']=next(row for row in selected['sweep'] if row['threshold']==1.)
        selected['disabled_reason']='Probability calibration lacks a positive or negative class; no discriminative calibration evidence'
    threshold=selected['selected']['threshold']
    atomic_json(dict(calibration=calibration,**selected),out/'calibration.json')
    atomic_save(dict(relation=relation.state_dict(),gate=gate.state_dict(),calibration=calibration,
                     threshold=threshold,protocol=protocol,admitted_for_runtime=False),out/'pilot.pt')
    # Smoke is TRAIN-only. Pilot Dev is read only once all choices are frozen.
    if a.evaluate_dev:
        if file_sha256(Path(a.dev)/'manifest.json')!=protocol['dev_manifest_sha']:raise ValueError('Dev changed')
        _,dev=load(a.dev)
        if {r['group'] for r in train}&{r['group'] for r in dev}:raise ValueError('TRAIN/Dev overlap')
        dev=prepare(dev,static)
        report,details=evaluate(dev,native,relation,gate,old_gate,device,sigma,calibration,threshold,old['threshold'])
        atomic_json(report,out/'internal_dev.json');atomic_json(details,out/'internal_dev_states.json')
    else:
        report,_=evaluate(cal,native,relation,gate,old_gate,device,sigma,calibration,threshold,old['threshold'])
        atomic_json(report,out/'train_smoke.json')
    if file_sha256(a.native_checkpoint)!=source_sha or file_sha256(a.old_pilot)!=old_sha or file_sha256(Path(a.stage1_cache)/'queries.jsonl')!=public_sha:raise ValueError('Frozen source changed')
    atomic_json(dict(complete=True,native_unchanged=True,old_pilot_unchanged=True,public_cache_unchanged=True,
                     dev_evaluated=a.evaluate_dev,checkpoint_sha=file_sha256(out/'pilot.pt'),admitted_for_runtime=False),out/'complete.json')
    print(json.dumps(dict(calibration=calibration,selection=selected['selected'],metrics=report)),flush=True)


if __name__=='__main__':main()
