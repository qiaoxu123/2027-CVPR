"""Existing static network/loss with shared frozen features and resumable training."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
import torch
from torch.nn import functional as F

from graphnav.candidate_geometry import FullCandidateDataset
from graphnav.polygon_ablation import geometry_view, geometry_groups, PAIR_MASKS
from graphnav.full_experiment import new_model
from graphnav.static_localization import cross_entropy_goal
from graphnav.stage1_runtime import retrieval_ids
from graphnav.token_language import FrozenBERTTokens
from graphnav.checkpointing import atomic_save, atomic_json, rng_state, restore_rng, file_sha256
from scripts.polygon_ablation_plan import OUT, BASE, query_key, canonical_job

MODEL_KEYS=('nodes','node_types','node_valid','entity_valid','prior','prior_buildings','rules','relation_fields','pair_geometry','grid_xy','edges','adjacency')


def event(**row):
    row['time']=time.time()
    print(json.dumps(row,ensure_ascii=False),flush=True)
    atomic_json(dict(status='Running',**row),OUT/'progress.json')
    with (OUT/'events.jsonl').open('a') as f:
        f.write(json.dumps(row,ensure_ascii=False)+'\n')


def model_for(plan, seed=None, frozen=False):
    if seed is not None:
        torch.manual_seed(seed)
    model=new_model(SimpleNamespace(**plan['args']),torch.device('cuda'))
    if frozen:
        state=torch.load(BASE,map_location='cpu',weights_only=True)
        model.load_state_dict({k[3:]:v for k,v in state['model'].items() if k.startswith('a3.')},strict=True)
        model.requires_grad_(False)
    else:
        # Same architecture/state keys, inactive branches in ALL representations.
        for module in (model.gnn.layer1,model.gnn.layer2,model.vision_proj,model.visual_score):
            module.requires_grad_(False)
    return model


def weight_hash(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        h.update(k.encode());h.update(v.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def encoder_for(plan):
    enc=FrozenBERTTokens(plan['args']['text_backbone'],device='cuda',max_length=96,cache_limit=100000)
    enc.load_cache(plan['args']['train_bert_cache'])
    first=enc.cache.copy()
    enc.load_cache(plan['args']['seen_bert_cache'])
    enc.cache.update(first)
    del first
    return enc


def make_dataset(plan, path):
    a=plan['args']
    return FullCandidateDataset(path,grid_size=28,max_landmarks=a['max_landmarks'],max_roads=a['max_roads'],
                                max_relations=a['max_relations'],prior_sigma_m=a['prior_sigma_m'],rule_sigma_m=a['rule_sigma_m'])


def prepare_store(plan, split, enc):
    cache=OUT/'.cache';cache.mkdir(exist_ok=True)
    target=cache/f'{split}_geometry.pt'
    source=plan['paths'][split]
    signature=dict(input_sha256=file_sha256(source),protocol=plan['version'],center=plan['feature_center'],
                   geometry_source_sha256=file_sha256(Path(__file__).resolve().parents[1]/'graphnav/polygon_ablation.py'))
    if target.exists():
        payload=torch.load(target,map_location='cpu',weights_only=True)
        if payload['signature']!=signature:
            raise ValueError('Feature cache signature differs')
        return Store(payload,plan)
    ds=make_dataset(plan,source);n=len(ds);a=plan['args'];capacity=ds.capacity
    tensors={};metadata=[];started=time.time()
    for i in range(n):
        item=ds[i];record=ds.records[i];graph=ds._prepared(i)[0]
        c=geometry_view(item,graph,record['world_bounds'],representation='centroid',scope='end_to_end',prior_sigma_m=a['prior_sigma_m'],rule_sigma_m=a['rule_sigma_m'])
        b=geometry_view(item,graph,record['world_bounds'],representation='bounding_box',scope='end_to_end',prior_sigma_m=a['prior_sigma_m'],rule_sigma_m=a['rule_sigma_m'])
        row={k:item[k] for k in ('nodes','node_types','node_valid','entity_valid','prior','prior_buildings','rules','relation_fields')}
        # Unused padded relation-node pairs are stored implicitly as zeros.
        row['pair_geometry']=item['pair_geometry'][:,:a['max_landmarks']]
        row['pair_bbox']=b['pair_geometry'][:,:a['max_landmarks']]
        for label,view in [('centroid',c),('bounding_box',b)]:
            for key in ('nodes','prior','prior_buildings','rules','relation_fields'):
                row[f'{key}__{label}']=view[key]
        row['target_xy']=item['target_xy'];row['map_sizes']=item['map_sizes']
        if i==0:
            tensors={k:torch.empty((n,*v.shape),dtype=v.dtype) for k,v in row.items()}
        for k,v in row.items():
            tensors[k][i].copy_(v)
        metadata.append(dict(index=i,query_id=query_key(record),episode_id=record['episode_id'],map_name=record['map_name'],
                             groups=geometry_groups(graph['entities'],graph['relations']),world_bounds=record['world_bounds'],entities=len(graph['entities']),
                             true_polyline_count=sum(e['kind']=='road' for e in graph['entities'])))
        if i%256==255:
            ds._prepared.cache_clear();ds._pair_cache.cache_clear()
        if i%1000==0:
            event(phase='feature_cache',split=split,completed=i,total=n,seconds=time.time()-started)
    # Cached input language tensors contain no labels; use existing frozen BERT
    # caches and compute only missing public instructions/names.
    language_path=cache/f'{split}_language.pt'
    if language_path.exists():
        language=torch.load(language_path,map_location='cpu',weights_only=True)
        assert language['input_sha256']==signature['input_sha256']
    else:
        tokens=torch.zeros((n,96,768),dtype=torch.float32)
        masks=torch.zeros((n,96),dtype=torch.bool)
        names=torch.empty((n,capacity,768),dtype=torch.float32)
        for start in range(0,n,64):
            end=min(start+64,n)
            rs=ds.records[start:end]
            text=[r['instruction'] for r in rs]
            # Name packing is representation-invariant and comes from the same
            # original graph. Only public source geometry/names determine it.
            packed=[ds._prepared(i)[0]['node_names'] for i in range(start,end)]
            t,m=enc.encode_tokens(text)
            tokens[start:end,:t.shape[1]]=t.cpu().float();masks[start:end,:m.shape[1]]=m.cpu()
            v=enc.encode([x for group in packed for x in group])
            names[start:end]=v.cpu().reshape(end-start,capacity,768).float()
            ds._prepared.cache_clear()
        language=dict(input_sha256=signature['input_sha256'],bert_revision=enc.revision,tokens=tokens,token_mask=masks,node_emb=names)
        atomic_save(language,language_path)
    tensors.update({k:language[k] for k in ('tokens','token_mask','node_emb')})
    payload=dict(signature=signature,tensors=tensors,metadata=metadata,grid_xy=torch.from_numpy(ds.grid_xy),capacity=capacity)
    # Language cache is separate to avoid duplicating multi-GB embeddings.
    atomic_save(dict(payload,tensors={k:v for k,v in tensors.items() if k not in ('tokens','token_mask','node_emb')}),target)
    atomic_json(dict(signature=signature,samples=n,language_cache=str(language_path),metadata=metadata),cache/f'{split}_manifest.json')
    event(phase='feature_cache_complete',split=split,samples=n,seconds=time.time()-started)
    return Store(payload,plan)


class Store:
    def __init__(self,payload,plan):
        self.plan=plan;self.metadata=payload['metadata'];self.capacity=payload['capacity']
        data=payload['tensors']
        if 'tokens' not in data:
            split=next(s for s,p in plan['paths'].items() if file_sha256(p)==payload['signature']['input_sha256'])
            language=torch.load(OUT/f'.cache/{split}_language.pt',map_location='cpu',weights_only=True)
            data.update({k:language[k] for k in ('tokens','token_mask','node_emb')})
        self.tensors={k:v.cuda() for k,v in data.items()}
        self.grid=payload['grid_xy'].cuda()
        self.n=len(self.metadata)

    def batch(self,ids,scope,representation):
        ids=ids.to('cuda');d=self.tensors;b=len(ids)
        keys=('nodes','node_types','node_valid','entity_valid','prior','prior_buildings','rules','relation_fields')
        x={k:d[k][ids] for k in keys}
        paths={'prior_to_centroid':('prior','prior_buildings'),'rules_to_centroid':('rules','relation_fields'),'nodes_to_centroid':('nodes',)}
        if representation in paths:
            for k in paths[representation]:x[k]=d[f'{k}__centroid'][ids]
        if scope=='end_to_end' and representation!='full_polygon':
            for k in ('nodes','prior','prior_buildings','rules','relation_fields'):
                x[k]=d[f'{k}__{representation}'][ids]
        p=d['pair_bbox' if representation=='bounding_box' else 'pair_geometry'][ids]
        mask=torch.tensor(PAIR_MASKS[representation],device='cuda',dtype=p.dtype)
        p=p*mask
        if representation=='no_inside_information':
            p[...,5]=p[...,5].abs()
        x['pair_geometry']=F.pad(p,(0,0,0,self.capacity-p.shape[-2]))
        x['grid_xy']=self.grid[None].expand(b,-1,-1)
        x['adjacency']=torch.zeros((b,self.capacity,self.capacity),device='cuda',dtype=torch.bool)
        x['edges']=torch.zeros((b,self.capacity,self.capacity,19),device='cuda')
        valid=d['token_mask'][ids];length=int(valid.sum(-1).max())
        return x,d['tokens'][ids,:length],valid[:,:length],d['node_emb'][ids],d['target_xy'][ids],d['map_sizes'][ids]


def forward(model,store,ids,scope,rep):
    inputs,tokens,mask,names,target,size=store.batch(ids,scope,rep)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        field=model(inputs,variant='geometry',tokens=tokens,token_mask=mask,node_emb=names)
    return field,target,size,inputs


@torch.no_grad()
def evaluate(model,store,scope,representation,indices=None,output=None):
    model.eval();ids=torch.arange(store.n) if indices is None else torch.as_tensor(indices)
    sums=np.zeros(4,np.float64);rows=[]
    if output and Path(output).exists():
        rows=[json.loads(l) for l in Path(output).read_text().splitlines()]
        assert [r['index'] for r in rows]==ids[:len(rows)].tolist()
        assert all(r['query_id']==store.metadata[r['index']]['query_id'] for r in rows)
        for r in rows:sums+=np.asarray([r[k] for k in ('R@1','R@4','R@8','NE')])
    completed=len(rows)
    for start in range(0,len(ids),16):
        part=ids[start:start+16]
        if start+len(part)<=completed:continue
        field,target,size,inputs=forward(model,store,part,scope,representation)
        chosen=retrieval_ids(field.belief_logits.float())
        xy=inputs['grid_xy'].gather(1,chosen[...,None].expand(-1,-1,2))
        distances=((xy-target[:,None])*size[:,None]).norm(dim=-1).cpu().numpy()
        for offset,(i,dist) in enumerate(zip(part.tolist(),distances)):
            if start+offset<completed:continue
            metrics={f'R@{k}':float(dist[:k].min()<=20) for k in (1,4,8)}
            metrics['NE']=float(dist[0]);sums+=np.asarray(list(metrics.values()))
            rows.append(dict(store.metadata[i],**metrics,top16_distances_m=dist.tolist(),top16_grid_ids=chosen[offset].cpu().tolist()))
        if output:
            output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
            with output.open('a') as f:
                for row in rows[completed:]:f.write(json.dumps(row)+'\n')
            completed=len(rows)
    summary=dict(samples=len(ids),**dict(zip(('R@1','R@4','R@8','NE'),(sums/len(ids)).tolist())))
    if output:
        output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
        atomic_json(summary,output.with_suffix('.metrics.json'))
    return summary


def train(plan,store,scope,rep,seed):
    name=canonical_job(scope,rep,seed);out=OUT/'training'/name;out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():
        done=json.loads((out/'complete.json').read_text());assert done['checkpoint_sha256']==file_sha256(out/'best.pt')
        event(phase='train_reuse_complete',job=name);return
    model=model_for(plan,seed=seed);initial=weight_hash(model)
    (out/'config.json').write_text(json.dumps(dict(scope=scope,representation=rep,seed=seed,plan_sha256=file_sha256(OUT/'frozen_plan.json'),initial_state_sha256=initial,
                                                 parameters=sum(p.numel() for p in model.parameters()),optimized_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),gat_message_passing=False),indent=2))
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=plan['lr'],weight_decay=plan['weight_decay'])
    members=json.loads((OUT/'train_dev_manifest.json').read_text())
    fit=torch.tensor([r['index'] for r in members if r['split']=='train_fit'])
    dev=[r['index'] for r in members if r['split']=='train_dev']
    epoch=1;next_batch=0;history=[];best=(-1.,float('-inf'));loss_sum=0.;seen=0
    if (out/'last.pt').exists():
        s=torch.load(out/'last.pt',map_location='cuda',weights_only=True)
        assert s['plan_sha256']==file_sha256(OUT/'frozen_plan.json')
        model.load_state_dict(s['model']);optimizer.load_state_dict(s['optimizer']);restore_rng(s['rng'])
        epoch=s['next_epoch'];next_batch=s['next_batch'];history=s['history'];best=tuple(s['best']);loss_sum=s['loss_sum'];seen=s['seen']
    def state(e,batch):
        return dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng_state(),next_epoch=e,next_batch=batch,
                    history=history,best=best,loss_sum=loss_sum,seen=seen,plan_sha256=file_sha256(OUT/'frozen_plan.json'),args=plan['args'],scope=scope,representation=rep,seed=seed)
    for e in range(epoch,plan['epochs']+1):
        if next_batch==0:
            torch.manual_seed(seed*1000003+e);loss_sum=0.;seen=0
        order=fit[torch.randperm(len(fit),generator=torch.Generator().manual_seed(seed*1000003+e))]
        model.train();started=time.time()
        for b,start in enumerate(range(0,len(order),plan['batch_size'])):
            if b<next_batch:continue
            ids=order[start:start+plan['batch_size']]
            field,target,_,_=forward(model,store,ids,scope,rep)
            loss=cross_entropy_goal(field.belief_logits,target,grid_size=28)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
            optimizer.step();loss_sum+=float(loss.detach())*len(ids);seen+=len(ids)
            if (b+1)%200==0:
                atomic_save(state(e,b+1),out/'last.pt')
                event(phase='training',job=name,epoch=e,batch=b+1,total_batches=(len(fit)+15)//16,loss=loss_sum/seen,seconds=time.time()-started)
        metrics=evaluate(model,store,scope,rep,indices=dev)
        history.append(dict(epoch=e,loss=loss_sum/seen,train_samples=seen,train_dev=metrics,seconds=time.time()-started,
                            data_order_sha256=hashlib.sha256(order.numpy().tobytes()).hexdigest()))
        key=(metrics['R@1'],-metrics['NE']);improved=key>best
        if improved:best=key
        payload=dict(model=model.state_dict(),args=plan['args'],scope=scope,representation=rep,seed=seed,epoch=e,selected_on=plan['selection'],plan_sha256=file_sha256(OUT/'frozen_plan.json'),initial_state_sha256=initial)
        atomic_save(payload,out/f'epoch_{e:02}.pt')
        if improved:atomic_save(payload,out/'best.pt')
        next_batch=0;loss_sum=0.;seen=0
        atomic_save(state(e+1,0),out/'last.pt');atomic_json(history,out/'history.json')
        event(phase='epoch_complete',job=name,**history[-1])
    atomic_json(dict(status='Done',epochs=plan['epochs'],checkpoint_sha256=file_sha256(out/'best.pt'),initial_state_sha256=initial,
                     training_queries=len(fit),dev_queries=len(dev),selected_on=plan['selection']),out/'complete.json')
    del model,optimizer
    torch.cuda.empty_cache()
