"""Independent two-stage training; private labels never enter model kwargs."""
from common import *
import argparse
import math
import time
import numpy as np
import torch
from model import LocalRelationNetwork
from labels import objective
from schema import SCHEMA_VERSION,validate_public,to_device


def audit_groups(groups):
    sets={k:{tuple(r['target_instance']) for r in rows} for k,rows in groups.items()}
    for a in sets:
        for b in sets:
            if a!=b and sets[a]&sets[b]:raise ValueError(f'Target instance leakage between {a} and {b}')
    return {s:dict(views=len(rows),instances=len(sets[s]),rank_positive_views=sum(bool(r['labels']['ranking'].sum()) for r in rows),
        informative_rank_views=sum(bool(r['labels']['ranking'].sum()) and int(r['labels']['rank_label_mask'].sum())>int((r['labels']['ranking']>0).sum()) for r in rows),
        height_pairs=sum(int(r['labels']['height_valid'].sum()) for r in rows)) for s,rows in groups.items()}


def load_groups(allow_partial=False):
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    if not manifest['complete'] and not allow_partial:raise ValueError('Partial feature cache: use --smoke only for diagnostic verification')
    from prepare import cache_signature
    if manifest['signature']!=cache_signature():raise ValueError('Cache implementation/config signature mismatch')
    groups={s:[] for s in ['train','calibration','dev']}
    for e in manifest['entries']:
        path=resolve_row_path(e['path'])
        if e.get('content_sha256') and sha(path)!=e['content_sha256']:raise ValueError('Cached row content changed')
        r=torch.load(path,map_location='cpu',weights_only=False)
        if r['signature']!=manifest['signature']:raise ValueError('Row signature mismatch')
        validate_public(r['public']);groups[e['split']].append(r)
    audit_groups(groups);return groups


def labels_on(row,device):return {k:v.to(device) for k,v in row['labels'].items()}


@torch.no_grad()
def evaluate_rows(model,rows,device='cpu',control='real'):
    model.eval();records=[]
    for r in rows:
        x=to_device(r['public'],device);y=labels_on(r,device);o=model(**x,control=control);loss,_=objective(o,y,True)
        positive=bool(y['ranking'].sum());n=len(o['scores']);s=r['public']['spatial'];item=dict(episode=r['episode'],step=r['step'],loss=float(loss),
            match_logit=float(o['match_logit']),match=float(y['match']),match_valid=bool(y['match_valid']),positive=positive)
        if positive and n:
            order=o['scores'].argsort(descending=True);new=int(order[0]);base=int(s['candidate_base'].argmax());rrank=(y['ranking'][order]>0).nonzero().flatten()
            pred=s['candidate_xy'][new].to(device)+o['offset'][new];error=float((pred-y['target_xy']).norm())
            item.update(rank1=bool(y['ranking'][new]>0),base_correct=bool(y['ranking'][base]>0),mrr=1/(int(rrank[0])+1),xy_error=error,
                base_xy_error=float((s['candidate_xy'][base].to(device)-y['target_xy']).norm()))
        records.append(item)
    ranked=[r for r in records if r['positive']]
    summary=dict(views=len(records),eligible=bool(ranked),ranked_views=len(ranked),rank1=float(np.mean([r['rank1'] for r in ranked])) if ranked else 0.,
        mrr=float(np.mean([r['mrr'] for r in ranked])) if ranked else 0.,xy_error=float(np.mean([r['xy_error'] for r in ranked])) if ranked else 1e6,
        new_errors=sum(r['base_correct'] and not r['rank1'] for r in ranked),repairs=sum(not r['base_correct'] and r['rank1'] for r in ranked),
        mean_loss=float(np.mean([r['loss'] for r in records])) if records else 0.)
    return summary,records


def select_epoch(metrics,best):
    if not metrics['eligible'] or metrics['new_errors']>0:return False
    key=lambda x:(x['rank1'],x['mrr'],-x['xy_error'],-x['epoch'])
    return best is None or key(metrics)>key(best)


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--steps',type=int,default=4)
    p.add_argument('--rank-epochs',type=int);p.add_argument('--offset-epochs',type=int);p.add_argument('--device',default='cuda');p.add_argument('--output',type=Path)
    p.add_argument('--cached-only',action='store_true',help='Train bundled frozen features without opening the external raw annotations; still verifies cache/config/model/source hashes')
    args=p.parse_args();torch.set_num_threads(2);torch.manual_seed(17);rng=np.random.default_rng(17)
    c=configuration();verify_sources(c,annotations=not args.cached_only);groups=load_groups(args.smoke);coverage=audit_groups(groups)
    if not groups['train']:raise ValueError('No training rows')
    output=args.output or ROOT/('runs/smoke' if args.smoke else 'runs/branch')
    if output.exists():raise ValueError(f'Refuse to overwrite existing training directory {output}')
    output.mkdir(parents=True);device=args.device;rgb_dim=groups['train'][0]['public']['network']['candidate_rgb'].shape[-1]
    model=LocalRelationNetwork(rgb_dim=rgb_dim).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
    instances={}
    for r in groups['train']:instances.setdefault(tuple(r['target_instance']),[]).append(r)
    buckets=list(instances.values());batch=c['batch_size'];per_epoch=max(1,math.ceil(len(groups['train'])/batch))
    rank_epochs=args.rank_epochs if args.rank_epochs is not None else c['rank_epochs'];offset_epochs=args.offset_epochs if args.offset_epochs is not None else c['offset_epochs']
    epochs=1 if args.smoke else rank_epochs+offset_epochs;steps=args.steps if args.smoke else per_epoch*epochs
    step=0;history=[];best=None;selected=None;started=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train();accum=[];use_offset=args.smoke or epoch>rank_epochs
        for _ in range(args.steps if args.smoke else per_epoch):
            optimizer.zero_grad();learning=c['lr']*min(1.,(step+1)/max(1,steps*.05))
            for group in optimizer.param_groups:group['lr']=learning
            for _ in range(batch):
                bucket=buckets[int(rng.integers(len(buckets)))];r=bucket[int(rng.integers(len(bucket)))]
                o=model(**to_device(r['public'],device));loss,_=objective(o,labels_on(r,device),use_offset)
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                (loss/batch).backward();accum.append(float(loss.detach()))
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();step+=1
        metrics,records=evaluate_rows(model,groups['dev'],device);metrics.update(epoch=epoch,step=step,train_loss=float(np.mean(accum)),offset_stage=use_offset)
        payload=dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},model_config=dict(rgb_dim=rgb_dim,hidden=256),
            schema_version=SCHEMA_VERSION,config_sha256=sha(ROOT/'configs/frozen.json'),cache_signature=json.loads((ROOT/'data/manifest.json').read_text())['signature'],
            implementation=implementation_hashes(),metrics=metrics,diagnostic_only=args.smoke,parameters=sum(p.numel() for p in model.parameters()))
        path=output/f'epoch_{epoch:03d}.pt';torch.save(payload,path)
        if select_epoch(metrics,best):best=metrics.copy();selected=path
        history.append(metrics);save(dict(status='training',epoch=epoch,epochs=epochs,step=step,seconds=time.perf_counter()-started,metrics=metrics),output/'status.json')
        print(json.dumps(metrics),flush=True)
    torch.save(payload,output/'last.pt')
    result=dict(coverage=coverage,history=history,selected_checkpoint=str(selected) if selected else None,selected_sha256=sha(selected) if selected else None,
        fallback_checkpoint=str(output/'last.pt'),diagnostic_only=args.smoke,feedback_eligible=bool(selected) and not args.smoke,seconds=time.perf_counter()-started,
        parameters=payload['parameters'],batch_implementation='8 independent observations accumulated before one optimizer step; no cross-observation attention',
        caveat='Trusted association labels require quality audit; no navigation efficacy claim')
    save(result,output/'training.json');save(dict(status='complete',**result),output/'status.json')


if __name__=='__main__':main()
