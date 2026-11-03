"""Seeded, resumable TRAIN-only position-head fitting; confirmation is separate.

The original A3/visual/controller weights and default navigation are untouched.
Targets supervise world-goal probabilities, not verified image-instance labels.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader,RandomSampler

from graphnav.checkpointing import atomic_json,atomic_save,file_sha256,rng_state,restore_rng
from graphnav.local_chain_position import LocalChainPositionHead,position_loss
from graphnav.visual_chain_position import VisualChainPositionHead,WorldLikelihoodPositionHead,visual_chain_loss
from graphnav.local_chain_raster import rasterized_goal_logprob
from graphnav.precision_preserve_loss import precision_preserve_loss

INPUTS=('prior_logits','local_uv','image_features','target_text','context_text','public_geometry','quality','local_valid')


def merge_fit_observations(base, extra):
    keys={r['task_key'] for r in base}
    if any(r['task_key'] not in keys for r in extra):
        raise ValueError('non-fit target in additional observations')
    frames=[r['frame_key'] for r in base+extra]
    if len(set(frames))!=len(frames):
        raise ValueError('duplicate fit observations')
    return base+extra


def records(cache,labels,subset,role_vectors=None,role_mode='full',allowed_frames=None):
    meta=json.loads((cache/'manifest.json').read_text())
    if not meta['complete']:raise ValueError('complete feature cache required')
    truth={r['frame_key']:r for r in map(json.loads,labels.read_text().splitlines()) if r['status']=='captured'}
    rows=[]
    for chunk in meta['chunks']:
        path=cache/chunk['file']
        if file_sha256(path)!=chunk['sha256']:raise ValueError('corrupt feature chunk')
        for r in torch.load(path,map_location='cpu',weights_only=True)['records']:
            if r['subset']!=subset:continue
            if allowed_frames is not None and r['frame_key'] not in allowed_frames:continue
            if r['frame_key'] not in truth:raise ValueError('missing same-subset label')
            label=truth[r['frame_key']]
            r.update(target_world=torch.tensor(label['target_world'][:2],dtype=torch.float32),category=label['category'])
            if role_vectors is not None:
                role=role_vectors[r['task_key']]
                r['full_text']=r['context_text'].clone()
                r['target_phrase']=role['target_phrase']
                if role_mode=='reference':r['context_text']=role['reference_vector'].clone()
            rows.append(r)
    if not rows:raise ValueError('empty '+subset)
    return rows


def collate(rows):
    out={k:torch.stack([r[k] for r in rows]) for k in (*INPUTS,'candidate_world','target_world')}
    for k in ('frame_key','task_key','category'):out[k]=[r[k] for r in rows]
    if 'target_phrase' in rows[0]:
        out['target_phrase']=[r['target_phrase'] for r in rows]
        out['full_text']=torch.stack([r['full_text'] for r in rows])
    return out


def batch_inputs(batch,device,control='real'):
    x={k:batch[k].to(device) for k in INPUTS}
    if control=='zero':x['image_features']=torch.zeros_like(x['image_features'])
    if control=='shuffle':
        image=x['image_features'].clone()
        for b in range(len(image)):
            ids=x['local_valid'][b].nonzero().flatten()
            image[b,ids]=x['image_features'][b,ids.flip(0)]
        x['image_features']=image
    if control=='no_geometry':x['public_geometry']=torch.zeros_like(x['public_geometry'])
    if control=='full_text':
        x['target_text']=batch.get('full_text',batch['context_text']).to(device)
        x['context_text']=x['target_text']
    return x


@torch.no_grad()
def evaluate(model,data,device,batch_size,control='real'):
    model.eval(); rows=[]
    for batch in DataLoader(data,batch_size=batch_size,collate_fn=collate):
        x=batch_inputs(batch,device,control);logits=model(**x)
        if getattr(model,'grid_output',False):
            logits=rasterized_goal_logprob(logits,x['local_uv'],x['local_valid'],global_cells=x['prior_logits'].shape[-1])
        chosen=logits.argmax(-1).cpu();base=batch['prior_logits'].argmax(-1)
        for i,choice in enumerate(chosen):
            target=batch['target_world'][i];world=batch['candidate_world'][i]
            ne=float((world[choice]-target).norm());base_ne=float((world[base[i]]-target).norm())
            rows.append(dict(frame_key=batch['frame_key'][i],task_key=batch['task_key'][i],category=batch['category'][i],
                             ne_m=ne,success=ne<=20.,baseline_ne_m=base_ne,baseline_success=base_ne<=20.,
                             chosen_local=bool(choice>=batch['prior_logits'].shape[-1])))
    grouped=defaultdict(list)
    for row in rows:grouped[row['task_key']].append(row)
    metric=dict(frames=len(rows),physical_targets=len(grouped),goal_r1=float(np.mean([r['success'] for r in rows])),
                goal_ne_m=float(np.mean([r['ne_m'] for r in rows])),
                baseline_goal_r1=float(np.mean([r['baseline_success'] for r in rows])),
                baseline_goal_ne_m=float(np.mean([r['baseline_ne_m'] for r in rows])),
                corrections=sum(not r['baseline_success'] and r['success'] for r in rows),
                harms=sum(r['baseline_success'] and not r['success'] for r in rows),
                local_selected=sum(r['chosen_local'] for r in rows),
                task_balanced_goal_r1=float(np.mean([np.mean([r['success'] for r in group]) for group in grouped.values()])))
    metric['net']=metric['corrections']-metric['harms']
    return dict(metrics=metric,rows=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',required=True,type=Path)
    p.add_argument('--fit-labels',type=Path)
    p.add_argument('--dev-labels',type=Path)
    p.add_argument('--confirm-labels',type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--device',default='cuda')
    p.add_argument('--max-batches',type=int,default=0)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--confirm-checkpoint',type=Path)
    p.add_argument('--architecture',choices=('legacy','visual-chain-v2','world-likelihood-v3'),default='legacy')
    p.add_argument('--role-vectors',type=Path)
    p.add_argument('--role-mode',choices=('full','reference'),default='full')
    p.add_argument('--extra-fit-cache',type=Path)
    p.add_argument('--extra-fit-labels',type=Path)
    p.add_argument('--loss-mode',choices=('standard','precision-preserve'),default='standard')
    p.add_argument('--fit-observation-filter',type=Path)
    p.add_argument('--steps-per-epoch',type=int,default=0)
    p.add_argument('--canonical-fit-order',action='store_true')
    a=p.parse_args()
    if a.steps_per_epoch<0:raise ValueError('nonnegative step budget required')
    if a.loss_mode=='precision-preserve' and a.architecture!='world-likelihood-v3':raise ValueError('precision loss requires V3')
    torch.manual_seed(a.seed);np.random.seed(a.seed)
    torch.set_num_threads(4)
    model={'legacy':LocalChainPositionHead,'visual-chain-v2':VisualChainPositionHead,'world-likelihood-v3':WorldLikelihoodPositionHead}[a.architecture]().to(a.device)
    role_vectors=torch.load(a.role_vectors,map_location='cpu',weights_only=True) if a.role_vectors else None
    if a.architecture in ('visual-chain-v2','world-likelihood-v3') and role_vectors is None:raise ValueError('explicit role vectors/phrases required')
    if a.confirm_checkpoint:
        if a.fit_observation_filter:raise ValueError('no fitting filter in confirmation')
        if a.extra_fit_cache or a.extra_fit_labels:raise ValueError('no augmentation in confirmation')
        if a.output.exists():raise FileExistsError('confirmation output already exists; no repeated evaluation')
        if not a.confirm_labels or a.fit_labels or a.dev_labels:raise ValueError('confirm labels only in confirmation mode')
        state=torch.load(a.confirm_checkpoint,map_location=a.device,weights_only=True)
        if state['feature_manifest_sha256']!=file_sha256(a.cache/'manifest.json'):raise ValueError('frozen cache mismatch')
        model.load_state_dict(state['model']);model.requires_grad_(False)
        data=records(a.cache,a.confirm_labels,'confirm',role_vectors,a.role_mode)
        result={control:evaluate(model,data,a.device,a.batch_size,control) for control in ('real','zero','shuffle','no_geometry','full_text')}
        atomic_json(dict(checkpoint_sha256=file_sha256(a.confirm_checkpoint),labels_sha256=file_sha256(a.confirm_labels),
                         split='TRAIN independent confirm maps',results=result,warning='Offline Goal prediction, NOT closed-loop SR or verified instance matching'),a.output)
        print(json.dumps({k:v['metrics'] for k,v in result.items()},indent=2));return
    if a.confirm_labels or not a.fit_labels or not a.dev_labels:raise ValueError('fit/dev labels only during training')
    if a.output.exists() and not a.resume:raise FileExistsError('do not overwrite experiment')
    a.output.mkdir(parents=True,exist_ok=True)
    allowed=set(json.loads(a.fit_observation_filter.read_text())['frame_keys']) if a.fit_observation_filter else None
    fit=records(a.cache,a.fit_labels,'fit',role_vectors,a.role_mode,allowed);dev=records(a.cache,a.dev_labels,'dev',role_vectors,a.role_mode)
    if bool(a.extra_fit_cache)!=bool(a.extra_fit_labels):raise ValueError('both extra fit inputs required')
    if a.extra_fit_cache:
        original=json.loads((a.cache/'manifest.json').read_text())
        added=json.loads((a.extra_fit_cache/'manifest.json').read_text())
        for key in ('checkpoint_sha256','public_catalog_sha256','static_queries_sha256','roles_sha256','camera_yaw_rad','global_grid','local_candidate_count','window_patch_sizes'):
            if original[key]!=added[key]:raise ValueError('additional feature protocol mismatch: '+key)
        if any(value!='fit' or original['map_assignment'].get(key)!='fit' for key,value in added['map_assignment'].items()):raise ValueError('non-fit additional map')
        fit=merge_fit_observations(fit,records(a.extra_fit_cache,a.extra_fit_labels,'fit',role_vectors,a.role_mode))
    if a.canonical_fit_order:fit.sort(key=lambda r:(r['map_name'],r['task_key'],r['step']))
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.001)
    start_epoch=0;history=[];best=(-1.,float('-inf'))
    fingerprint={key:file_sha256(path) for key,path in [('feature_manifest_sha256',a.cache/'manifest.json'),('fit_labels_sha256',a.fit_labels),('dev_labels_sha256',a.dev_labels)]}
    if a.role_vectors:fingerprint['role_vectors_sha256']=file_sha256(a.role_vectors)
    if a.fit_observation_filter:fingerprint['fit_observation_filter_sha256']=file_sha256(a.fit_observation_filter)
    if a.loss_mode=='precision-preserve':fingerprint['loss_code_sha256']=file_sha256(Path(__file__).resolve().parents[1]/'graphnav/precision_preserve_loss.py')
    if a.extra_fit_cache:
        fingerprint.update(extra_feature_manifest_sha256=file_sha256(a.extra_fit_cache/'manifest.json'),extra_fit_labels_sha256=file_sha256(a.extra_fit_labels))
    if a.architecture in ('visual-chain-v2','world-likelihood-v3'):
        code_root=Path(__file__).resolve().parents[1]
        fingerprint.update(model_code_sha256=file_sha256(code_root/'graphnav/visual_chain_position.py'),
                           raster_code_sha256=file_sha256(code_root/'graphnav/local_chain_raster.py'),
                           trainer_code_sha256=file_sha256(__file__))
    if a.resume:
        state=torch.load(a.output/'last.pt',map_location=a.device,weights_only=True)
        if any(state[k]!=v for k,v in fingerprint.items()):raise ValueError('resume input SHA mismatch')
        if any(state['config'][key]!=getattr(a,key) for key in ('lr','seed','batch_size','max_batches')):raise ValueError('resume configuration mismatch')
        if state.get('architecture','legacy')!=a.architecture or state['config'].get('role_mode','full')!=a.role_mode:raise ValueError('resume architecture/role mismatch')
        if state['config'].get('canonical_fit_order',False)!=a.canonical_fit_order:raise ValueError('resume observation ordering changed')
        if state['config'].get('steps_per_epoch',0)!=a.steps_per_epoch:raise ValueError('resume update budget changed')
        if state['config'].get('loss_mode','standard')!=a.loss_mode:raise ValueError('resume loss changed')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);restore_rng(state['rng'])
        start_epoch=state['epoch'];history=state['history'];best=tuple(state['best'])
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    atomic_json(dict(config=config,**fingerprint,fit_frames=len(fit),dev_frames=len(dev),
                     trainable_parameters=sum(p.numel() for p in model.parameters()),selection='TRAIN dev Goal R1 then NE',
                     supervision='world-position only; no pixel instance labels'),a.output/'config.json')
    started=time.perf_counter();first_gradient=None
    for epoch in range(start_epoch,a.epochs):
        model.train();losses=[];eligible=0;examples=0;color_rows=0;fine_car_rows=0;preservation_rows=0
        generator=torch.Generator().manual_seed(a.seed+epoch)
        sampler=RandomSampler(fit,replacement=True,num_samples=a.steps_per_epoch*a.batch_size,generator=generator) if a.steps_per_epoch else None
        loader=DataLoader(fit,batch_size=a.batch_size,shuffle=sampler is None,sampler=sampler,generator=generator,collate_fn=collate)
        for step,batch in enumerate(loader):
            if a.max_batches and step>=a.max_batches:break
            optimizer.zero_grad(set_to_none=True)
            public=batch_inputs(batch,a.device)
            if a.architecture in ('visual-chain-v2','world-likelihood-v3'):
                logits,visual=model(**public,return_visual=True)
                loss_fn=precision_preserve_loss if a.loss_mode=='precision-preserve' else visual_chain_loss
                terms=loss_fn(model,logits,visual,public,batch['candidate_world'].to(a.device),batch['target_world'].to(a.device),batch['category'],batch['target_phrase'])
                color_rows+=terms['color_rows']
                fine_car_rows+=terms.get('fine_car_rows',0);preservation_rows+=terms.get('preservation_rows',0)
            else:
                logits=model(**public)
                terms=position_loss(logits,batch['candidate_world'].to(a.device),batch['target_world'].to(a.device),global_cells=784)
            if not torch.isfinite(terms['total']):raise FloatingPointError('nonfinite position loss')
            terms['total'].backward()
            if first_gradient is None:
                first_gradient={name:float(p.grad.norm()) if p.grad is not None else None for name,p in model.named_parameters()}
                if not any(v and v>0 for k,v in first_gradient.items() if k.startswith('image.')):raise RuntimeError('no visual gradient')
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step();losses.append(float(terms['total'].detach()));eligible+=terms['local_supervised'];examples+=len(batch['target_world'])
        validation=evaluate(model,dev,a.device,a.batch_size)
        entry=dict(epoch=epoch+1,loss=float(np.mean(losses)),local_supervised=eligible,color_contrastive_rows=color_rows,examples=examples,
                   fine_car_rows=fine_car_rows,preservation_rows=preservation_rows,
                   seconds=time.perf_counter()-started,dev=validation['metrics'])
        history.append(entry)
        key=(entry['dev']['goal_r1'],-entry['dev']['goal_ne_m']);is_best=key>best
        if is_best:best=key
        state=dict(model=model.state_dict(),architecture=a.architecture,optimizer=optimizer.state_dict(),rng=rng_state(),epoch=epoch+1,
                   config=config,history=history,best=best,**fingerprint)
        atomic_save(state,a.output/'last.pt')
        if is_best:atomic_save(state,a.output/'best.pt')
        atomic_json(dict(history=history,first_batch_gradient_norms=first_gradient,
                         peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                         changed_parameter_tensors=sum(not torch.equal(v.detach().cpu(),initial[k]) for k,v in model.state_dict().items())),a.output/'training.json')
        print(json.dumps(entry),flush=True)
    print(json.dumps(dict(best_checkpoint_sha256=file_sha256(a.output/'best.pt'),seconds=time.perf_counter()-started)),flush=True)


if __name__=='__main__':main()
