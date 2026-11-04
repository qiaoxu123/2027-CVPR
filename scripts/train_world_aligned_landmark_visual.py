"""World-aligned loss correction; same fit budget, reused TRAIN Dev regression only."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from graphnav.checkpointing import atomic_json,atomic_save,file_sha256
from graphnav.geometry_guided_data import load_landmark_data,batch_landmarks
from graphnav.geometry_guided_visual import LandmarkVisualBelief,visual_world_beliefs
from graphnav.world_aligned_landmark_loss import world_landmark_visual_loss as landmark_visual_loss


def eligible(row):
    return bool((row['labels']==1).any() and (row['labels']==0).any())


def replacements(rows,same_type):
    result={}
    for r in rows:
        candidates=[x for x in rows if x['metadata']['query_key']!=r['metadata']['query_key'] and
            eligible(x) and
            x['metadata']['frame']['image_sha256']!=r['metadata']['frame']['image_sha256'] and
            (not same_type or x['metadata']['public_type']==r['metadata']['public_type'])]
        candidates.sort(key=lambda x:hashlib.sha256(('0:'+r['metadata']['sample_key']+':'+x['metadata']['sample_key']).encode()).hexdigest())
        if candidates:result[r['metadata']['sample_key']]=candidates[0]
    return result


@torch.inference_mode()
def evaluate(model,rows,control='real'):
    model.eval();out=[]
    replacement=replacements(rows,control=='same_type_rgb') if control in ('wrong_rgb','same_type_rgb') else None
    for start in range(0,len(rows),16):
        chosen=rows[start:start+16]
        if replacement is not None:chosen=[r for r in chosen if r['metadata']['sample_key'] in replacement]
        if control=='wrong_polygon':chosen=[r for r in chosen if 'wrong_polygon_geometry' in r['teacher']]
        if not chosen:continue
        source=[replacement[r['metadata']['sample_key']] for r in chosen] if replacement is not None else None
        b=batch_landmarks(chosen,control=control,replacement=source)
        scores=model(image_features=b['image_features'],name_text=b['name_text'],type_text=b['type_text'],valid=b['valid'])
        fields=visual_world_beliefs(b['geometry_logits'],b['local_uv'],scores,b['valid'],b['projection_supported'])
        g=b['geometry_logits'].argmax(-1).cpu();f=fields['fused'].argmax(-1).cpu();v=fields['raw'].argmax(-1).cpu()
        local=scores.masked_fill(~b['valid'],float('-inf')).argmax(-1).cpu()
        for i,r in enumerate(chosen):
            gd=float(r['region_distance'][g[i]]);fd=float(r['region_distance'][f[i]]);vd=float(r['region_distance'][v[i]])
            rank=scores[i].masked_fill(~b['valid'][i],float('-inf')).argsort(descending=True).cpu()
            positives=r['labels'][rank]==1;positions=positives.nonzero().flatten()
            out.append(dict(sample_key=r['metadata']['sample_key'],query_key=r['metadata']['query_key'],frame_key=r['metadata']['frame_key'],
                map_name=r['metadata']['map_name'],type=r['metadata']['public_type'],eligible=eligible(r),
                public_projection_supported=r['projection_supported'],raw_has_view=bool(fields['has_view'][i]),
                weak_top1=bool(r['labels'][local[i]]==1),weak_top5=bool(positives[:5].any()),
                weak_mrr=1./(int(positions[0])+1) if len(positions) else 0.,
                geometry_ne=gd,geometry_success=gd<=20,fused_ne=fd,fused_success=fd<=20,
                raw_ne=vd,raw_success=vd<=20,raw_entropy=float(fields['raw'][i].softmax(-1).mul(fields['raw'][i].log_softmax(-1).nan_to_num()).sum().neg()),
                visual_energy_range=float((fields['energy'][i].max()-fields['energy'][i].min()).cpu()),
                selected_rgb_source=source[i]['metadata']['sample_key'] if source is not None else None))
    supported=[r for r in out if r['public_projection_supported'] and r['raw_has_view']]
    weak=[r for r in out if r['eligible']]
    metrics=dict(samples=len(out),named_groups=len({r['query_key'] for r in out}),weak_eligible=len(weak),
        weak_top1=sum(r['weak_top1'] for r in weak),weak_top5=sum(r['weak_top5'] for r in weak),
        weak_mrr=float(np.mean([r['weak_mrr'] for r in weak])) if weak else None,
        geometry_correct=sum(r['geometry_success'] for r in out),fused_correct=sum(r['fused_success'] for r in out),
        geometry_ne=float(np.mean([r['geometry_ne'] for r in out])),fused_ne=float(np.mean([r['fused_ne'] for r in out])),
        fused_corrected=sum(r['fused_success'] and not r['geometry_success'] for r in out),
        fused_harmed=sum(not r['fused_success'] and r['geometry_success'] for r in out),
        raw_supported=len(supported),raw_correct=sum(r['raw_success'] for r in supported),
        raw_ne=float(np.mean([r['raw_ne'] for r in supported])) if supported else None)
    return dict(metrics=metrics,rows=out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('data','teacher','catalog','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--smoke-steps',type=int);p.add_argument('--resume',action='store_true');p.add_argument('--evaluate',action='store_true')
    a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0);rng=random.Random(0)
    source={str(p):file_sha256(p) for p in [a.data/'manifest.json',a.teacher/'manifest.json',a.catalog,Path(__file__),
        Path('graphnav/geometry_guided_visual.py'),Path('graphnav/geometry_guided_data.py'),Path('graphnav/world_aligned_landmark_loss.py')]}
    data=load_landmark_data(a.data,a.teacher,a.catalog);fit=[r for r in data if r['metadata']['subset']=='fit' and eligible(r)]
    groups=defaultdict(list)
    for r in fit:
        key=r['metadata']['query_key']
        if not any(x['metadata']['frame']['image_sha256']==r['metadata']['frame']['image_sha256'] for x in groups[key]):groups[key].append(r)
    keys=sorted(groups);steps_per_epoch=math.ceil(len(fit)/16)
    model=LandmarkVisualBelief().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    if a.evaluate:
        sealed=json.loads((a.output/'freeze.json').read_text())
        if file_sha256(a.output/'final.pt')!=sealed['checkpoint_sha256']:raise ValueError('Unfrozen checkpoint')
        state=torch.load(a.output/'final.pt',weights_only=True,map_location='cpu')
        if state['source_sha256']!=source:raise ValueError('Evaluation source changed')
        model.load_state_dict(state['model']);baseline=LandmarkVisualBelief().cuda()
        controls=('real','zero_rgb','shuffle_rgb','wrong_rgb','same_type_rgb','wrong_polygon')
        results={}
        for subset in ('dev',):
            chosen=[r for r in data if r['metadata']['subset']==subset]
            results[subset]={'native':evaluate(baseline,chosen)}
            for control in controls:results[subset][control]=evaluate(model,chosen,control)
        atomic_json(results,a.output/'evaluation.json')
        print(json.dumps({s:{c:x['metrics'] for c,x in r.items()} for s,r in results.items()},indent=2));return
    if a.output.exists() and not a.resume:raise FileExistsError('Preserve previous training')
    a.output.mkdir(parents=True,exist_ok=True);start=0;history=[];gradients={};seen=set()
    if a.resume:
        last=torch.load(a.output/'last.pt',weights_only=True,map_location='cpu')
        if last['source_sha256']!=source:raise ValueError('Resume source changed')
        model.load_state_dict(last['model']);optimizer.load_state_dict(last['optimizer']);start=last['step'];history=last['history'];gradients=last['gradients'];seen=set(last['seen_frames'])
        rng.setstate(last['rng']);torch.set_rng_state(last['torch_rng']);torch.cuda.set_rng_state_all(last['cuda_rng'])
    config=dict(seed=0,epochs=12,batch=16,lr=.001,steps_per_epoch=steps_per_epoch,
        distillation_domain='784-world-grid conditional KL + global G-to-GV preservation',
        dev_role='reused TRAIN development regression, not fresh confirmation',confirmation_evaluation=False,
        fit_samples=len(fit),fit_names=len(groups),multi_view_names=sum(len(g)>1 for g in groups.values()),
        trainable_parameters=sum(p.numel() for p in model.parameters()),loss_weights=dict(region=1,distill=.25,contrast=.1),
        source_sha256=source,no_formal_val_or_test=True,backbone_frozen=True)
    atomic_json(config,a.output/'config.json');total=a.smoke_steps or 12*steps_per_epoch
    started=time.perf_counter();model.train();torch.cuda.reset_peak_memory_stats()
    for step in range(start,total):
        rows=[]
        for _ in range(8):
            group=groups[rng.choice(keys)]
            rows+=rng.sample(group,2) if len(group)>1 else [group[0],group[0]]
        seen.update(r['metadata']['frame_key'] for r in rows)
        b=batch_landmarks(rows)
        if step<2:b['image_features'].requires_grad_(True)
        loss=landmark_visual_loss(model,b)
        if not torch.isfinite(loss['total']):raise ValueError('Nonfinite loss')
        optimizer.zero_grad();loss['total'].backward()
        if step==1:
            gradients={k:float(p.grad.abs().sum()) if p.grad is not None else None for k,p in model.named_parameters()}
            gradients['image_input']=float(b['image_features'].grad.float().abs().sum())
            if not gradients['image_input']>0 or any(v is None or not math.isfinite(v) for v in gradients.values()):raise ValueError('Gradient disconnected/nonfinite')
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        if (step+1)%steps_per_epoch==0 or step+1==total:
            h=dict(step=step+1,epoch=(step+1)/steps_per_epoch,seconds=time.perf_counter()-started,
                **{k:float(v.detach()) if torch.is_tensor(v) else float(v) for k,v in loss.items()})
            history.append(h)
            state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step+1,history=history,gradients=gradients,
                source_sha256=source,rng=rng.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),seen_frames=sorted(seen))
            atomic_save(state,a.output/'last.pt');atomic_json(history,a.output/'history.json');print(json.dumps(h),flush=True)
    if not a.smoke_steps:
        atomic_save(state,a.output/'final.pt')
        atomic_json(dict(checkpoint_sha256=file_sha256(a.output/'final.pt'),steps=total,seconds=time.perf_counter()-started,
            peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,fit_frames_seen=len(seen),dev_used_for_fitting=False,
            confirmation_used_for_fitting=False,source_sha256=source),a.output/'freeze.json')


if __name__=='__main__':main()
