"""Joint Geometry/PGVB/direct-target training on every eligible TRAIN fit query."""
import argparse
from collections import Counter
import json
import random
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from graphnav.checkpointing import atomic_save, atomic_json, file_sha256, rng_state, restore_rng
from graphnav.direct_target_data import load_model, batch_inputs


def sample_rows(tasks, labels, cache):
    samples=[]
    for task in tasks:
        item=torch.load(cache/f"{task['index']:05d}.pt",map_location='cpu',weights_only=True)
        for obs in item['observations']:
            samples.append(dict(query=item['query'], **obs, task=task,
                                label=labels[task['index']]['target_world']))
    return samples


def coordinates(samples, inputs):
    bounds=torch.tensor([s['task']['world_bounds'] for s in samples],device='cuda')
    grid=bounds[:,None,:2]+inputs['static_inputs']['grid_xy']*(bounds[:,None,2:]-bounds[:,None,:2])
    target=torch.tensor([s['label'] for s in samples],device='cuda')
    distance=(grid-target[:,None]).norm(dim=-1)
    return distance


@torch.no_grad()
def evaluate(model, rows, batch_size=8):
    model.eval(); result=[]
    for start in range(0,len(rows),batch_size):
        batch=rows[start:start+batch_size]; inputs=batch_inputs(batch)
        model.reset_episode()  # a new independent query batch is not one episode
        with torch.autocast('cuda',dtype=torch.bfloat16): out=model.forward_direct_target(**inputs)
        d=coordinates(batch,inputs)
        for i,s in enumerate(batch):
            row=dict(index=s['task']['index'],step=s['step'],category=s['task']['structured']['target']['category'],
                     relation=s['task']['anchors'][0]['predicate'],RGB=s['observed'] is not None)
            for method,key in [('Geometry','geometry_field'),('Relation','relation_field'),('RGB','field')]:
                logits=out[key].belief_logits.float(); top=logits[i].argmax()
                row[method]=dict(NE=float(d[i,top]),R20=bool(d[i,top]<=20),
                    mass20=float((logits[i].softmax(-1)*(d[i]<=20)).sum()))
            result.append(row)
    metrics={method:dict(NE=float(np.mean([r[method]['NE'] for r in result])),
                        R20=float(np.mean([r[method]['R20'] for r in result])),
                        mass20=float(np.mean([r[method]['mass20'] for r in result])))
             for method in ('Geometry','Relation','RGB')}
    for left,right in [('Geometry','Relation'),('Relation','RGB')]:
        metrics[right]['corrected_vs_'+left]=sum(not r[left]['R20'] and r[right]['R20'] for r in result)
        metrics[right]['damaged_vs_'+left]=sum(r[left]['R20'] and not r[right]['R20'] for r in result)
    return metrics,result


def main():
    p=argparse.ArgumentParser()
    for k in ('inventory','cache','checkpoint','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--epochs',type=int,default=5);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--smoke-updates',type=int,default=0);p.add_argument('--resume',action='store_true')
    p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    torch.set_num_threads(4);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    tasks=json.loads((a.inventory/'public_tasks.json').read_text());labels=json.loads((a.inventory/'private_labels.json').read_text())
    if len(list(a.cache.glob('[0-9]*.pt')))!=len(tasks):raise RuntimeError('Full eligible cache must be complete before fitting')
    fit=sample_rows([t for t in tasks if t['subset']=='fit'],labels,a.cache)
    dev=sample_rows([t for t in tasks if t['subset']=='dev'],labels,a.cache)
    source=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    model=load_model(source);model.requires_grad_(False)
    for module in (model.geometry,model.pgvb,model.direct_target):module.requires_grad_(True)
    # Geometry's unused GAT message-passing layers do not run in its original
    # `geometry` variant. They remain in checkpoint for compatibility.
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW([dict(params=model.geometry.parameters(),lr=1e-5),
        dict(params=model.pgvb.parameters(),lr=3e-5),
        dict(params=model.direct_target.parameters(),lr=1e-4)],weight_decay=.001)
    config=dict(seed=a.seed,epochs=a.epochs,batch_size=a.batch_size,fit_queries=len({s['task']['index'] for s in fit}),
        fit_states=len(fit),dev_queries=len({s['task']['index'] for s in dev}),dev_states=len(dev),
        source_sha256=file_sha256(a.checkpoint),inventory_sha256=file_sha256(a.inventory/'public_tasks.json'),
        label_sha256=file_sha256(a.inventory/'private_labels.json'),cache_manifest_sha256=file_sha256(a.cache/'manifest.json'),
        code_sha256={k:file_sha256(k) for k in [__file__,'graphnav/direct_target.py','graphnav/direct_target_data.py','graphnav/unified_pgvb.py']},
        trainable_parameters=sum(p.numel() for p in params),foundation_encoders='frozen cached BERT/SigLIP2',
        trainable_core=['integrated Geometry','PGVB','DirectTargetHead'],controller='frozen historical Stage1',
        losses='CE15m(Geometry)+CE15m(Relation)+CE15m(RGB); no pixel GT',
        precision='CUDA BF16 with FP32 losses',gradient_clip=5.)
    a.output.mkdir(parents=True,exist_ok=True)
    epoch=0;offset=0;updates=0;best=float('inf');history=[];coverage=set();grad_audit={}
    if a.resume:
        saved=torch.load(a.output/'last.pt',map_location='cpu',weights_only=True)
        if saved['training_config']!=config:raise ValueError('Resume inputs/config changed')
        model.load_state_dict(saved['model'],strict=True);opt.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
        epoch=saved['epoch'];offset=saved['offset'];updates=saved['updates'];best=saved['best'];history=saved['history'];coverage=set(saved['fit_coverage'])
        atomic_json(dict(updates=updates,epoch=epoch,optimizer_entries=len(opt.state)),a.output/'resume_proof.json')
    elif (a.output/'last.pt').exists():raise FileExistsError('Refuse overwrite; use resume or new output')
    atomic_json(config,a.output/'config.json')
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter();first_weights={k:v.detach().clone() for k,v in model.named_parameters() if k in ['geometry.geometry_gain','pgvb.image.0.weight','direct_target.predicate.weight']}
    def save(name,ready):
        atomic_save(dict(format='unified-pgvb-direct-target-v1',model=model.state_dict(),config=source['config'],
            direct_target_ready=ready,training_config=config,optimizer=opt.state_dict(),rng=rng_state(),
            epoch=epoch,offset=offset,updates=updates,best=best,history=history,fit_coverage=sorted(coverage)),a.output/name)
    with (a.output/'steps.jsonl').open('a') as stream:
        while epoch<a.epochs:
            order=np.random.default_rng(a.seed+epoch).permutation(len(fit)).tolist()
            model.train();epoch_losses=[];epoch_start=time.perf_counter()
            while offset<len(order):
                batch=[fit[i] for i in order[offset:offset+a.batch_size]]
                torch.cuda.synchronize();t=time.perf_counter();inputs=batch_inputs(batch)
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):out=model.forward_direct_target(**inputs)
                distances=coordinates(batch,inputs);soft=(-distances.square()/450.).softmax(-1)
                loss=sum(-(soft*field.belief_logits.float().log_softmax(-1)).sum(-1).mean()
                         for field in (out['geometry_field'],out['relation_field'],out['field']))
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                loss.backward()
                if not grad_audit:
                    for prefix in ('geometry.','pgvb.','direct_target.'):
                        active={k:float(p.grad.abs().sum()) for k,p in model.named_parameters()
                                if k.startswith(prefix) and p.grad is not None}
                        if not active or sum(active.values())<=0:raise RuntimeError('Core gradient missing: '+prefix)
                        if not all(torch.isfinite(p.grad).all() for k,p in model.named_parameters()
                                   if k.startswith(prefix) and p.grad is not None):raise FloatingPointError('Gradient nonfinite')
                        grad_audit[prefix]=active
                    atomic_json(grad_audit,a.output/'gradient_audit.json')
                norm=torch.nn.utils.clip_grad_norm_(params,5.,error_if_nonfinite=True);opt.step()
                updates+=1;offset+=len(batch);coverage.update(s['task']['index'] for s in batch)
                torch.cuda.synchronize();record=dict(update=updates,epoch=epoch,offset=offset,
                    loss=float(loss),gradient_norm=float(norm),wall_ms=(time.perf_counter()-t)*1000,
                    peak_gpu_GiB=torch.cuda.max_memory_allocated()/2**30)
                stream.write(json.dumps(record)+'\n');epoch_losses.append(float(loss))
                if updates%25==0:stream.flush();print(json.dumps(record),flush=True)
                if updates%100==0:save('last.pt',False)
                if a.smoke_updates and updates>=a.smoke_updates:
                    save('last.pt',False)
                    changes={k:float((dict(model.named_parameters())[k]-v).abs().sum()) for k,v in first_weights.items()}
                    if not all(v>0 for v in changes.values()):raise RuntimeError('Core optimizer update missing')
                    atomic_json(dict(updates=updates,seconds=time.perf_counter()-started,changes=changes,
                                     finite_loss=True,complete_training=False),a.output/'smoke.json');return
            epoch+=1;offset=0;model.direct_target_ready=True
            metrics,rows=evaluate(model,dev,a.batch_size)
            rec=dict(epoch=epoch,updates=updates,fit_loss=float(np.mean(epoch_losses)),dev=metrics,
                     epoch_seconds=time.perf_counter()-epoch_start,elapsed_seconds=time.perf_counter()-started)
            history.append(rec);print(json.dumps(rec),flush=True)
            improved=metrics['RGB']['NE']<best;best=min(best,metrics['RGB']['NE'])
            save('last.pt',True)
            if improved:
                save('best.pt',True);atomic_json(rows,a.output/'best_dev_predictions.json')
            atomic_json(history,a.output/'history.json')
    changes={k:float((dict(model.named_parameters())[k]-v).abs().sum()) for k,v in first_weights.items()}
    atomic_json(dict(complete_training=True,epochs=epoch,updates=updates,fit_queries_seen=len(coverage),
        expected_fit_queries=config['fit_queries'],seconds=time.perf_counter()-started,
        peak_gpu_GiB=torch.cuda.max_memory_allocated()/2**30,changes=changes,
        best_checkpoint_sha256=file_sha256(a.output/'best.pt')),a.output/'complete.json')


if __name__=='__main__':main()
