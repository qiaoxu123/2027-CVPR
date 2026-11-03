"""Matched TRAIN transport pilots; only new reference-transport weights train."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader,RandomSampler
from graphnav.checkpointing import atomic_json,atomic_save,file_sha256,rng_state,restore_rng
from graphnav.reference_evidence_transport import ReferenceTransportPositionHead
from graphnav.visual_chain_position import WorldLikelihoodPositionHead
from scripts.train_local_chain_position import records,collate,batch_inputs


def public_inputs(batch,device):
    x=batch_inputs(batch,device)
    cells=x['prior_logits'].shape[-1]
    return dict(**x,global_world=batch['candidate_world'][:,:cells].to(device),local_world=batch['candidate_world'][:,cells:].to(device),instruction_text=batch['full_text'].to(device))


def goal_loss(logits,world,target):
    positive=(world-target[:,None,:2]).norm(dim=-1)<=20;eligible=positive.any(-1)
    if not eligible.any():return logits.sum()*0
    lp=logits[eligible].log_softmax(-1);mask=positive[eligible]
    mass=-torch.logsumexp(lp.masked_fill(~mask,float('-inf')),-1).mean()
    pos=lp.masked_fill(~mask,float('-inf')).amax(-1);neg=lp.masked_fill(mask,float('-inf')).amax(-1)
    return mass+.1*torch.relu(.2+neg-pos).mean()


def wrong_frame_features(data):
    result={};partners={}
    for row in data:
        options=[s for s in data if s['map_name']==row['map_name'] and s['task_key']!=row['task_key'] and s['local_valid'].any()]
        if not options:partners[row['frame_key']]=None;continue
        width=max(float(row['quality'][0,3]),1e-6)
        def key(s):return (abs(np.log(max(float(s['quality'][0,3]),1e-6)/width)),hashlib.sha256(s['frame_key'].encode()).hexdigest())
        other=min(options,key=key);valid=other['local_valid'].nonzero().flatten()
        ids=torch.arange(len(row['local_valid']));xy=torch.stack((ids%12,ids//12),-1).float();donor_xy=xy[valid]
        nearest=torch.cdist(xy,donor_xy).argmin(-1)
        result[row['frame_key']]=other['image_features'][valid[nearest]].clone()
        partners[row['frame_key']]=dict(donor=other['frame_key'],public_width_ratio=float(other['quality'][0,3])/width)
    return result,partners


@torch.no_grad()
def evaluate(model,data,control='real',donors=None):
    model.eval();output=[]
    for batch in DataLoader(data,batch_size=8,collate_fn=collate):
        x=public_inputs(batch,'cuda');extra={}
        if control=='wrong-frame':
            extra['transport_image_features']=torch.stack([donors.get(k,batch['image_features'][i]) for i,k in enumerate(batch['frame_key'])]).cuda()
        packed,energy=model(**x,control='real' if control=='wrong-frame' else control,return_evidence=True)
        chosen=packed[:,:784].argmax(-1).cpu();baseline=model.base(**batch_inputs(batch,'cuda'))[:,:784].argmax(-1).cpu()
        for i,c in enumerate(chosen):
            target=batch['target_world'][i];world=batch['candidate_world'][i]
            ne=float((world[c]-target).norm());old=float((world[baseline[i]]-target).norm())
            output.append(dict(frame_key=batch['frame_key'][i],task_key=batch['task_key'][i],category=batch['category'][i],ne_m=ne,success=ne<=20,
                baseline_ne_m=old,baseline_success=old<=20,evidence_min=float(energy[i].min()),evidence_max=float(energy[i].max())))
    metrics=dict(frames=len(output),physical_targets=len({x['task_key'] for x in output}),goal_r1=float(np.mean([x['success'] for x in output])),goal_ne_m=float(np.mean([x['ne_m'] for x in output])),
        corrected=sum(x['success'] and not x['baseline_success'] for x in output),harmed=sum(not x['success'] and x['baseline_success'] for x in output),
        baseline_goal_r1=float(np.mean([x['baseline_success'] for x in output])),baseline_ne_m=float(np.mean([x['baseline_ne_m'] for x in output])))
    return dict(metrics=metrics,rows=output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=('local','global','constant'),required=True)
    p.add_argument('--epochs',type=int,default=3);p.add_argument('--steps-per-epoch',type=int,default=159);p.add_argument('--resume',action='store_true');a=p.parse_args()
    smoke=a.steps_per_epoch==2
    if a.steps_per_epoch not in (2,159) or not 1<=a.epochs<=3 or (not smoke and a.epochs!=3):raise ValueError('fixed three epochs or bounded smoke')
    if a.output.exists() and not a.resume:raise FileExistsError('preserve old experiment')
    if a.resume and (a.output/'status.json').exists() and json.loads((a.output/'status.json').read_text())['status']=='complete':raise ValueError('already complete; no repeated development selection')
    r=a.root;torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0)
    cache=r/'chain_train400_features';source=r/'world_v3_envelope_s0/best.pt';roles_path=r/'reference_vectors_v2.pt';filter_path=r/'public_deployment_filter.json'
    fingerprint={str(path):file_sha256(path) for path in [source,roles_path,filter_path,cache/'manifest.json',r/'fixed_yaw_train400_s1/labels_fit.jsonl',r/'fixed_yaw_train400_s1/labels_dev.jsonl',Path(__file__),Path('graphnav/reference_evidence_transport.py'),Path('graphnav/visual_chain_position.py'),Path('graphnav/evidence_reachability.py'),Path('scripts/train_local_chain_position.py')]}
    initial=torch.load(source,map_location='cpu',weights_only=True)
    if initial['feature_manifest_sha256']!=file_sha256(cache/'manifest.json') or initial['role_vectors_sha256']!=file_sha256(roles_path):raise ValueError('source feature protocol differs')
    roles=torch.load(roles_path,map_location='cpu',weights_only=True);allowed=set(json.loads(filter_path.read_text())['frame_keys'])
    fit=records(cache,r/'fixed_yaw_train400_s1/labels_fit.jsonl','fit',roles,'reference',allowed)
    if len(fit)!=900:raise ValueError('same full 900 fit states required')
    base=WorldLikelihoodPositionHead();base.load_state_dict(initial['model']);model=ReferenceTransportPositionHead(base,mode=a.mode).cuda()
    total=torch.zeros(3,1024);count=0
    for row in fit:
        total+=row['image_features'][row['local_valid']].float().sum(0);count+=int(row['local_valid'].sum())
    model.constant_features.copy_((total/max(1,count)).cuda())
    optimizer=torch.optim.AdamW(model.transport.parameters(),lr=.001,weight_decay=.001)
    config=dict(root=str(r),output=str(a.output),mode=a.mode,seed=0,batch_size=8,lr=.001,epochs=a.epochs,steps_per_epoch=a.steps_per_epoch)
    history=[];start=0;score_gradient=None;image_gradient=None
    if a.resume:
        state=torch.load(a.output/'last.pt',map_location='cpu',weights_only=True)
        if state['fingerprint']!=fingerprint:raise ValueError('resume source/data changed')
        for k in config:
            if k!='epochs' and config[k]!=state['config'][k]:raise ValueError('resume config changed')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);restore_rng(state['rng']);history=state['history'];start=state['epoch'];score_gradient=state['score_gradient'];image_gradient=state['image_gradient']
    a.output.mkdir(parents=True,exist_ok=True)
    def checkpoint(epoch):return dict(architecture='global-reference-transport-v1',model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng_state(),epoch=epoch,history=history,
        config=config,fingerprint=fingerprint,score_gradient=score_gradient,image_gradient=image_gradient,base_checkpoint_sha256=file_sha256(source))
    if not a.resume:atomic_save(checkpoint(0),a.output/'last.pt')
    first=collate(fit[:8]);x=public_inputs(first,'cuda')
    if start==0:
        with torch.no_grad():
            old=model.base(**batch_inputs(first,'cuda'));new=model(**x)
        if not torch.equal(old,new):raise RuntimeError('zero initialization changed original prediction')
    atomic_json(dict(config=config,fingerprint=fingerprint,fit_frames=len(fit),trainable_parameters=sum(x.numel() for x in model.transport.parameters()),normal_inference_unchanged=True),a.output/'config.json')
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for epoch in range(start,a.epochs):
        model.train();losses=[];generator=torch.Generator().manual_seed(epoch);sampler=RandomSampler(fit,replacement=True,num_samples=a.steps_per_epoch*8,generator=generator)
        for batch in DataLoader(fit,batch_size=8,sampler=sampler,generator=generator,collate_fn=collate):
            optimizer.zero_grad(set_to_none=True);x=public_inputs(batch,'cuda');packed=model(**x)
            loss=goal_loss(packed[:,:784],x['global_world'],batch['target_world'].cuda())
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite Goal loss')
            loss.backward()
            if any(p.grad is not None for p in model.base.parameters()):raise RuntimeError('original head received gradient')
            for param,key in [(model.transport.evidence.weight,'score'),(model.transport.image.weight,'image')]:
                if param.grad is None or not torch.isfinite(param.grad).all():raise RuntimeError('missing/nonfinite new-module gradient')
                value=float(param.grad.norm())
                if value>0:
                    if key=='score' and score_gradient is None:score_gradient=value
                    if key=='image' and image_gradient is None:image_gradient=value
            torch.nn.utils.clip_grad_norm_(model.transport.parameters(),5.);optimizer.step();losses.append(float(loss.detach()))
        history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),steps=a.steps_per_epoch,seconds=time.perf_counter()-started));atomic_save(checkpoint(epoch+1),a.output/'last.pt')
        atomic_json(dict(history=history,score_gradient=score_gradient,image_gradient=image_gradient,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30),a.output/'training.json');print(json.dumps(history[-1]),flush=True)
    if any(not torch.equal(v.cpu(),initial['model'][k]) for k,v in model.base.state_dict().items()):raise RuntimeError('original parameters changed')
    if score_gradient is None or image_gradient is None:raise RuntimeError('no effective visual evidence learning')
    if smoke:atomic_json(dict(status='smoke_complete',development_evaluated=False),a.output/'status.json');return
    atomic_save(checkpoint(a.epochs),a.output/'final.pt');model.eval()
    dev=records(cache,r/'fixed_yaw_train400_s1/labels_dev.jsonl','dev',roles,'reference')
    if {x['map_name'] for x in fit}&{x['map_name'] for x in dev}:raise ValueError('fit/development map overlap')
    donors,partners=wrong_frame_features(dev)
    results={control:evaluate(model,dev,control,donors) for control in ('real','zero-image','shuffle-image','mean-image','wrong-frame','zero-relation')}
    if any(x['ne_m']!=x['baseline_ne_m'] for x in results['zero-image']['rows']):raise RuntimeError('zero image did not recover frozen base')
    fit_result=evaluate(model,fit)
    atomic_json(dict(checkpoint_sha256=file_sha256(a.output/'final.pt'),results=results,fit=fit_result,wrong_frame_partners=partners,base_frozen_exact=True,
        warning='TRAIN development only; not verified reference-instance matching or navigation SR'),a.output/'evaluation.json')
    atomic_json(dict(status='complete',base_frozen_exact=True),a.output/'status.json');print(json.dumps(dict(status='complete',results={k:v['metrics'] for k,v in results.items()})),flush=True)


if __name__=='__main__':main()
