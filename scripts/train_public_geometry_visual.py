"""Fixed-budget TRAIN map/visual comparisons; dev opens only after suite freeze."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from graphnav.checkpointing import atomic_json,atomic_save,file_sha256
from graphnav.public_geometry_visual import PublicGeometryVisualHead
from graphnav.visual_chain_position import WorldLikelihoodPositionHead
from scripts.train_local_chain_position import records,collate,batch_inputs
from scripts.train_reference_evidence_transport import goal_loss


def load_fit(root):
    oldroles=torch.load(root/'reference_vectors_v2.pt',weights_only=True)
    newroles=torch.load(root/'scoped_between_train294_reference_vectors_v1.pt',weights_only=True)
    oldallowed=set(json.loads((root/'public_deployment_filter.json').read_text())['frame_keys'])
    newallowed=set(json.loads((root/'scoped_between_train294_public_filter_v1.json').read_text())['frame_keys'])
    old=records(root/'chain_train400_features',root/'fixed_yaw_train400_s1/labels_fit.jsonl','fit',oldroles,'reference',oldallowed)
    new=records(root/'scoped_between_train294_features_v1',root/'scoped_between_train294_observations_v1/labels_private.jsonl','fit',newroles,'reference',newallowed)
    assert len(old)==900 and len(new)==774 and not ({x['task_key'] for x in old}&{x['task_key'] for x in new})
    return old+new


def model_inputs(rows,geometry):
    batch=collate(rows);x=batch_inputs(batch,'cuda')
    x['world_geometry']=torch.stack([geometry[r['frame_key']] for r in rows]).cuda()
    x['instruction_text']=batch['full_text'].cuda()
    return x,batch


def evaluate(model,data,geometry,control='real'):
    model.eval();rows=[]
    with torch.inference_mode():
        for start in range(0,len(data),8):
            selected=data[start:start+8];x,batch=model_inputs(selected,geometry)
            out,base,m,v=model(**x,control=control,return_base=True)
            chosen=out[:,:784].argmax(-1).cpu();previous=base[:,:784].argmax(-1).cpu();a0=batch['prior_logits'].argmax(-1)
            for i,r in enumerate(selected):
                distance=(batch['candidate_world'][i,:784]-batch['target_world'][i]).norm(dim=-1)
                rows.append(dict(frame_key=r['frame_key'],task_key=r['task_key'],category=r['category'],ne_m=float(distance[chosen[i]]),
                    success=bool(distance[chosen[i]]<=20),base_ne_m=float(distance[previous[i]]),base_success=bool(distance[previous[i]]<=20),
                    a0_ne_m=float(distance[a0[i]]),a0_success=bool(distance[a0[i]]<=20),
                    map_range=float(m[i].max()-m[i].min()),visual_range=float(v[i].max()-v[i].min())))
    metrics=dict(frames=len(rows),targets=len({r['task_key'] for r in rows}),correct=sum(r['success'] for r in rows),
        r1=float(np.mean([r['success'] for r in rows])),ne_m=float(np.mean([r['ne_m'] for r in rows])),
        corrected=sum(r['success'] and not r['base_success'] for r in rows),harmed=sum(not r['success'] and r['base_success'] for r in rows),
        baseline_correct=sum(r['base_success'] for r in rows),baseline_ne_m=float(np.mean([r['base_ne_m'] for r in rows])))
    return dict(metrics=metrics,rows=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','geometry','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--mode',choices=('map','unconditioned','conditioned'),required=True)
    p.add_argument('--steps',type=int,default=630);p.add_argument('--resume',action='store_true')
    p.add_argument('--evaluate',action='store_true');p.add_argument('--freeze-manifest',type=Path)
    a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0)
    if a.steps not in (2,4,630):raise ValueError('fixed smoke/resume or 3x210 updates only')
    r=a.root;source=r/'world_v3_envelope_s0/best.pt';geometry=torch.load(a.geometry,weights_only=True)
    fingerprint={str(path):file_sha256(path) for path in [source,a.geometry,r/'reference_vectors_v2.pt',r/'scoped_between_train294_reference_vectors_v1.pt',
        r/'chain_train400_features/manifest.json',r/'scoped_between_train294_features_v1/manifest.json',
        r/'fixed_yaw_train400_s1/labels_fit.jsonl',r/'scoped_between_train294_observations_v1/labels_private.jsonl',
        r/'public_deployment_filter.json',r/'scoped_between_train294_public_filter_v1.json',Path(__file__),Path('graphnav/public_geometry_visual.py')]}
    base=WorldLikelihoodPositionHead();initial=torch.load(source,map_location='cpu',weights_only=True);base.load_state_dict(initial['model'])
    model=PublicGeometryVisualHead(base,mode=a.mode).cuda()
    trainable=[x for x in model.parameters() if x.requires_grad]
    optimizer=torch.optim.AdamW(trainable,lr=.001,weight_decay=.001)
    if a.evaluate:
        if not a.freeze_manifest:raise ValueError('suite freeze before dev evaluation required')
        freeze=json.loads(a.freeze_manifest.read_text());path=a.output/'final.pt'
        if freeze['checkpoints'].get(str(path))!=file_sha256(path):raise ValueError('checkpoint not frozen')
        state=torch.load(path,map_location='cpu',weights_only=True)
        if state['fingerprint']!=fingerprint:raise ValueError('source changed')
        model.load_state_dict(state['model'])
        roles=torch.load(r/'reference_vectors_v2.pt',weights_only=True)
        dev=records(r/'chain_train400_features',r/'fixed_yaw_train400_s1/labels_dev.jsonl','dev',roles,'reference')
        controls=('real',) if a.mode=='map' else ('real','zero','shuffle')
        results={c:evaluate(model,dev,geometry,c) for c in controls}
        atomic_json(results,a.output/'dev_final.json');print(json.dumps({c:v['metrics'] for c,v in results.items()}),flush=True);return
    if a.output.exists() and not a.resume:raise FileExistsError(a.output)
    fit=load_fit(r);start=0;history=[];gradients={}
    if a.resume:
        state=torch.load(a.output/'last.pt',map_location='cpu',weights_only=True)
        if state['fingerprint']!=fingerprint or state['mode']!=a.mode:raise ValueError('resume source/mode changed')
        start=state['step'];model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);history=state['history'];gradients=state['gradients']
        torch.set_rng_state(state['rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    else:
        a.output.mkdir(parents=True)
        x,_=model_inputs(fit[:8],geometry)
        with torch.no_grad():out,old,m,v=model(**x,return_base=True)
        if not torch.equal(out,old) or m.abs().max()!=0 or v.abs().max()!=0:raise ValueError('strict same-call initial identity failed')
    config=dict(mode=a.mode,seed=0,steps=a.steps,lr=.001,batch=8,fit_frames=len(fit),fit_targets=len({x['task_key'] for x in fit}),
        trainable_parameters=sum(x.numel() for x in trainable),output=str(a.output),fingerprint=fingerprint)
    atomic_json(config,a.output/'config.json');started=time.perf_counter();model.train()
    for step in range(start,a.steps):
        ids=torch.randint(len(fit),(8,)).tolist();selected=[fit[i] for i in ids];x,batch=model_inputs(selected,geometry)
        out,old,_,_=model(**x,return_base=True)
        loss=goal_loss(out[:,:784],batch['candidate_world'][:,:784].cuda(),batch['target_world'].cuda())
        lp=out[:,:784].log_softmax(-1);target=old[:,:784].detach().softmax(-1)
        loss=loss+.05*F.kl_div(lp,target,reduction='batchmean')
        if not torch.isfinite(loss):raise ValueError('nonfinite loss')
        optimizer.zero_grad();loss.backward()
        if step>=1 and not gradients:
            gradients={k:float(p.grad.abs().sum()) if p.grad is not None else None for k,p in model.named_parameters() if k in ('image.weight','geometry.weight','map_score.weight','visual_score.weight')}
        torch.nn.utils.clip_grad_norm_(trainable,5.);optimizer.step()
        if (step+1)%210==0 or step+1==a.steps:
            for k,v in initial['model'].items():
                if not torch.equal(model.base.state_dict()[k].cpu(),v.cpu()):raise ValueError('frozen base modified')
            torch.cuda.synchronize();history.append(dict(step=step+1,loss=float(loss),loop_seconds=time.perf_counter()-started,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30))
            state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step+1,mode=a.mode,history=history,gradients=gradients,
                rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),fingerprint=fingerprint)
            atomic_save(state,a.output/'last.pt');atomic_json(history,a.output/'history.json')
            print(json.dumps(history[-1]),flush=True)
    if a.steps==630:
        atomic_save(state,a.output/'final.pt')
        atomic_json(evaluate(model,fit,geometry),a.output/'fit_final.json')
    atomic_json(dict(complete=True,step=a.steps,gradients=gradients,last_sha256=file_sha256(a.output/'last.pt')),a.output/'status.json')

if __name__=='__main__':main()
