"""Original V3, same augmented TRAIN data: coarse position vs soft instance loss.

New queries receive fresh A0/BERT/SigLIP text/map fields; image-only descriptors
are shared. GT association and confidence exist only in collate/loss/evaluation.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import time

import numpy as np
import torch
from torch.utils.data import DataLoader,RandomSampler

from graphnav.candidate_geometry import relative_entity_features
from graphnav.checkpointing import atomic_json,atomic_save,file_sha256,rng_state,restore_rng
from graphnav.instance_ranking_loss import noncar_position_auxiliary
from graphnav.retarget_training_data import retarget_overrides,weak_instance_mask
from graphnav.uncertain_instance_loss import uncertain_instance_loss
from graphnav.visual_chain_position import WorldLikelihoodPositionHead,visual_chain_loss
from scripts.train_local_chain_position import records,collate as old_collate,batch_inputs,evaluate


def collate(rows):
    out=old_collate(rows)
    if 'weak_instance_labels' in rows[0]:
        out['weak_instance_labels']=torch.stack([r['weak_instance_labels'] for r in rows])
        out['soft_preference']=torch.tensor([r['soft_preference'] for r in rows],dtype=torch.float32)
    return out


def attach_retarget_data(base,fit_labels,teacher,query_cache):
    proof=json.loads((teacher/'summary.json').read_text())
    for name,filename in [('public_links','public_links.json'),('private_labels','loss_labels_private.pt')]:
        if proof[name+'_sha256']!=file_sha256(teacher/filename):raise ValueError('teacher input changed')
    query_proof=json.loads((query_cache/'summary.json').read_text())
    if not query_proof['complete'] or query_proof['public_inputs_sha256']!=file_sha256(query_cache/'public_inputs.pt'):
        raise ValueError('new-language public inputs missing/corrupt')
    if query_proof['public_tasks_sha256']!=proof['public_tasks_sha256'] or query_proof['public_queries_sha256']!=proof['public_queries_sha256']:
        raise ValueError('new-query text provenance mismatch')
    links=json.loads((teacher/'public_links.json').read_text())
    labels=torch.load(teacher/'loss_labels_private.pt',map_location='cpu',weights_only=True)
    queries=torch.load(query_cache/'public_inputs.pt',map_location='cpu',weights_only=True)
    source={row['frame_key']:row for row in base}
    truth={row['frame_key']:row for row in map(json.loads,fit_labels.read_text().splitlines())}
    by_frame=defaultdict(dict);strength={}
    for link in links:
        private=labels[link['sample_key']];oid=str(private['physical_group'][1])
        positive=(private['weak_instance_labels']==1).nonzero().flatten().tolist()
        if positive:by_frame[link['source_frame_key']][oid]=positive
        strength[(link['source_frame_key'],oid)]=private['soft_preference']
    for row in base:
        oid=str(truth[row['frame_key']]['physical_group'][1])
        row['weak_instance_labels']=weak_instance_mask(by_frame[row['frame_key']],oid,len(row['local_valid'])) if row['category']=='Car' else torch.full_like(row['local_valid'],-1,dtype=torch.int8)
        row['soft_preference']=strength.get((row['frame_key'],oid),.5)
    added=[];skipped=[]
    for link in links:
        original=source[link['source_frame_key']];query=queries[link['task_key']];private=labels[link['sample_key']]
        words=set(re.findall(r'[a-z]+',query['target_phrase'].lower()))
        if not words&{'car','cars','vehicle','vehicles','van','vans','truck','bus','jeep','suv','automobile','taxi'}:
            skipped.append(link['task_key']);continue
        if original['map_name']!=query['map_name']:raise ValueError('retarget map mismatch')
        bounds=np.asarray(query['world_bounds'],np.float32);axis=(np.arange(28,dtype=np.float32)+.5)/28
        gx,gy=np.meshgrid(axis,axis,indexing='xy');global_world=bounds[:2]+np.stack((gx.ravel(),gy.ravel()),-1)*(bounds[2:]-bounds[:2])
        if not np.allclose(original['candidate_world'][:784].numpy(),global_world,atol=1e-3):raise ValueError('new-query world grid differs from image source')
        geometry=np.zeros((len(original['local_valid']),31),np.float32)
        entities=query['public_entities']
        if entities:
            pair=relative_entity_features(original['candidate_world'][784:].numpy(),entities,len(entities))
            geometry=np.concatenate((pair.mean(1),pair.min(1),pair.max(1),np.full((len(pair),1),len(entities)/16,np.float32)),-1)
        row=retarget_overrides(original,query,torch.from_numpy(geometry).half())
        row.update(frame_key=link['sample_key'],source_frame_key=link['source_frame_key'],task_key=link['task_key'],category='Car',
            target_world=torch.tensor(private['target_world'],dtype=torch.float32),
            weak_instance_labels=private['weak_instance_labels'],soft_preference=private['soft_preference'],subset='fit')
        added.append(row)
    return base+added,dict(base_frames=len(base),retarget_frames=len(added),unique_new_queries=len({r['task_key'] for r in added}),
        skipped_role_queries=sorted(set(skipped)),soft_instance_rows=sum(r['soft_preference']>.5 and (r['weak_instance_labels']==1).any().item() and (r['weak_instance_labels']==0).any().item() for r in base+added))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('cache','fit-labels','dev-labels','filter','role-vectors','teacher','public-inputs','warm-start','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--loss-mode',choices=('coarse','soft-instance'),required=True)
    parser.add_argument('--epochs',type=int,default=3);parser.add_argument('--steps-per-epoch',type=int,default=159)
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    smoke=args.steps_per_epoch==2
    if args.steps_per_epoch not in (2,159) or not 1<=args.epochs<=3 or (not smoke and args.epochs!=3):raise ValueError('fixed three epochs or bounded smoke')
    if args.output.exists() and not args.resume:raise FileExistsError('preserve prior model')
    if args.resume and (args.output/'status.json').exists() and json.loads((args.output/'status.json').read_text())['status']=='complete':raise ValueError('already completed, no repeat development selection')
    torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0)
    fingerprint={str(path):file_sha256(path) for path in (args.cache/'manifest.json',args.fit_labels,args.dev_labels,args.filter,args.role_vectors,
        args.teacher/'summary.json',args.teacher/'public_links.json',args.teacher/'loss_labels_private.pt',args.public_inputs/'summary.json',
        args.public_inputs/'public_inputs.pt',args.warm_start,Path(__file__),Path('graphnav/retarget_training_data.py'),
        Path('graphnav/uncertain_instance_loss.py'),Path('graphnav/visual_chain_position.py'),Path('graphnav/instance_ranking_loss.py'),Path('scripts/train_local_chain_position.py'))}
    source=torch.load(args.warm_start,map_location='cpu',weights_only=True)
    if source['architecture']!='world-likelihood-v3' or source['feature_manifest_sha256']!=file_sha256(args.cache/'manifest.json') or source['role_vectors_sha256']!=file_sha256(args.role_vectors):raise ValueError('warm-start protocol mismatch')
    teacher_proof=json.loads((args.teacher/'summary.json').read_text())
    query_proof=json.loads((args.public_inputs/'summary.json').read_text())
    cache_proof=json.loads((args.cache/'manifest.json').read_text())
    if teacher_proof['feature_manifest_sha256']!=source['feature_manifest_sha256'] or source['fit_observation_filter_sha256']!=file_sha256(args.filter):raise ValueError('teacher/Goal sampling protocol differs')
    if query_proof['checkpoint_sha256']!=cache_proof['checkpoint_sha256'] or query_proof['catalog_sha256']!=cache_proof['public_catalog_sha256']:raise ValueError('retarget A0/map weights differ from source observations')
    model=WorldLikelihoodPositionHead().cuda();model.load_state_dict(source['model'],strict=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    roles=torch.load(args.role_vectors,map_location='cpu',weights_only=True)
    allowed=set(json.loads(args.filter.read_text())['frame_keys'])
    base=records(args.cache,args.fit_labels,'fit',roles,'reference',allowed)
    fit,dataset_summary=attach_retarget_data(base,args.fit_labels,args.teacher,args.public_inputs)
    config={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()};config.update(role_mode='reference',seed=0,batch_size=8,lr=.001)
    history=[];start=0;fine_gradient=None
    if args.resume:
        state=torch.load(args.output/'last.pt',map_location='cpu',weights_only=True)
        if state['fingerprint']!=fingerprint:raise ValueError('resume source/input SHA differs')
        for key in config:
            if key not in ('epochs','resume') and config[key]!=state['config'][key]:raise ValueError('resume config differs: '+key)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);restore_rng(state['rng'])
        history=state['history'];start=state['epoch'];fine_gradient=state['fine_visual_gradient']
    args.output.mkdir(parents=True,exist_ok=True)
    def checkpoint(epoch):
        return dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng_state(),epoch=epoch,history=history,
            fine_visual_gradient=fine_gradient,config=config,fingerprint=fingerprint,architecture='world-likelihood-v3',
            feature_manifest_sha256=file_sha256(args.cache/'manifest.json'),role_vectors_sha256=file_sha256(args.role_vectors),
            warm_start_checkpoint_sha256=file_sha256(args.warm_start),selection='fixed final epoch, no development choice')
    if not args.resume:atomic_save(checkpoint(0),args.output/'last.pt')
    atomic_json(dict(config=config,fingerprint=fingerprint,dataset=dataset_summary,normal_inference_changed=False),args.output/'config.json')
    print(json.dumps(dict(dataset=dataset_summary,status='training')),flush=True)
    started=time.perf_counter()
    for epoch in range(start,args.epochs):
        model.train();losses=[];fine_losses=[];effective=0
        generator=torch.Generator().manual_seed(epoch)
        sampler=RandomSampler(fit,replacement=True,num_samples=args.steps_per_epoch*8,generator=generator)
        for batch in DataLoader(fit,batch_size=8,sampler=sampler,generator=generator,collate_fn=collate):
            optimizer.zero_grad(set_to_none=True);public=batch_inputs(batch,'cuda')
            logits,visual=model(**public,return_visual=True)
            terms=visual_chain_loss(model,logits,visual,public,batch['candidate_world'].cuda(),batch['target_world'].cuda(),batch['category'],batch['target_phrase'])
            fine,count=uncertain_instance_loss(visual,batch['weak_instance_labels'].cuda(),public['local_valid'],batch['soft_preference'].cuda())
            loss=terms['total']
            if args.loss_mode=='soft-instance':
                distance=(batch['candidate_world'][:,784:].cuda()-batch['target_world'][:,None].cuda()).norm(dim=-1)
                noncar=noncar_position_auxiliary(visual,distance,public['local_valid'],batch['category'])
                # Retain official global Goal loss. Replace only coarse Car
                # appearance/color positives by uncertainty-aware identities.
                loss=loss-.5*terms['visual']-.1*terms['contrast']+.5*noncar+.5*fine
                if count and fine_gradient is None and float(fine.detach())>0:
                    grad=torch.autograd.grad(fine,model.image.weight,retain_graph=True)[0]
                    fine_gradient=float(grad.norm())
                    if not torch.isfinite(grad).all() or fine_gradient<=0:raise RuntimeError('no effective private-instance visual gradient')
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite grounding loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            losses.append(float(loss.detach()));fine_losses.append(float(fine.detach()));effective+=count
        entry=dict(epoch=epoch+1,loss=float(np.mean(losses)),soft_instance_loss=float(np.mean(fine_losses)),effective_instance_presentations=effective,
            steps=args.steps_per_epoch,seconds=time.perf_counter()-started)
        history.append(entry);atomic_save(checkpoint(epoch+1),args.output/'last.pt')
        atomic_json(dict(history=history,fine_visual_gradient=fine_gradient,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30),args.output/'training.json')
        print(json.dumps(entry),flush=True)
    if smoke:atomic_json(dict(status='smoke_complete',development_evaluated=False),args.output/'status.json');return
    model.eval();atomic_save(checkpoint(args.epochs),args.output/'final.pt')
    clone=WorldLikelihoodPositionHead();clone.load_state_dict(torch.load(args.output/'final.pt',map_location='cpu',weights_only=True)['model'],strict=True)
    dev=records(args.cache,args.dev_labels,'dev',roles,'reference')
    if {r['map_name'] for r in dev}&{r['map_name'] for r in fit}:raise ValueError('fitting/development map leakage')
    baseline=WorldLikelihoodPositionHead().cuda().eval();baseline.load_state_dict(source['model'])
    reference=evaluate(baseline,dev,'cuda',8)
    results={control:evaluate(model,dev,'cuda',8,control) for control in ('real','zero','shuffle','no_geometry','full_text')}
    atomic_json(dict(checkpoint_sha256=file_sha256(args.output/'final.pt'),frozen_baseline=reference,results=results,
        navigation_run=False,warning='TRAIN development Goal scores, not actual instance accuracy or SR'),args.output/'development.json')
    atomic_json(dict(status='complete',strict_original_deployment_load=True),args.output/'status.json')
    print(json.dumps(dict(status='complete',results={key:value['metrics'] for key,value in results.items()})),flush=True)


if __name__=='__main__':main()
