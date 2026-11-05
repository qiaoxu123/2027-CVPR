"""TRAIN-only cached FP32 Relation experiment, isolated from normal navigation."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from graphnav.checkpointing import atomic_json,atomic_save,file_sha256
from graphnav.direct_target_data import load_model,batch_inputs,relation_geometry
from graphnav.direct_target_language import PREDICATES
from graphnav.direct_target_runtime import bind_public_anchors
from graphnav.landmark_region_supervision import named_union
from graphnav.candidate_geometry import FullCandidateDataset,collate_full
from graphnav.full_experiment import _batch_inputs
from graphnav.token_language import FrozenBERTTokens
from scripts.analyze_instruction_landmark_association import PublicNames
from scripts.relation_counterfactual_metrics import replacement_span
from scripts.polygon_relation_prior import relation_prior,WORDS


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=6);p.add_argument('--resume',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
    if (a.output/'last.pt').exists() and not a.resume:
        raise FileExistsError('Preserve previous experiment; choose new output or --resume')
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    source=a.source/'training_aligned/best.pt';source_sha=file_sha256(source)
    bundle=torch.load(source,map_location='cpu',weights_only=True);model=load_model(bundle).eval().requires_grad_(False)
    tasks=json.loads((a.source/'inventory_validated/public_tasks.json').read_text())
    catalog=json.loads((a.data_root/'cityrefer/objects.json').read_text());names=PublicNames(catalog)
    records=[json.loads(l) for l in (a.source/'cache_aligned/public_static.jsonl').read_text().splitlines()]
    assert all(r['target_xy']==[0.,0.] for r in records)
    args=bundle['config']['geometry_args'];cachefile=a.output/'features.pt';started=time.perf_counter()
    if cachefile.exists():
        cache=torch.load(cachefile,map_location='cpu',weights_only=True)
        assert cache['source_sha']==source_sha
        natural=cache['natural'];counter=cache['counter']
    else:
        bert=FrozenBERTTokens(args['text_backbone'],device='cuda',max_length=args['max_text_length'])
        natural=[];counter=[]
        with torch.no_grad():
            for task in tasks:
                item=torch.load(a.source/'cache_aligned'/f'{task["index"]:05d}.pt',map_location='cpu',weights_only=True)
                inputs=batch_inputs([dict(query=item['query'])],visual=False)
                model.reset_episode();base=model._geometry_field(inputs['static_inputs'],inputs['tokens'],inputs['token_mask'],inputs['node_emb'])
                bounds=np.asarray(task['world_bounds']);world=bounds[:2]+inputs['static_inputs']['grid_xy'][0].cpu().numpy()*(bounds[2:]-bounds[:2])
                bound,_=bind_public_anchors(task['anchors'],catalog[task['map_name']]);poly=named_union(bound[0]['contours'])
                original=bound[0]['predicate'];group=task['map_name']+'::'+','.join(bound[0]['landmark']['public_ids'])
                def pack(base,tok,mask,geometry,word,logits,region,text):
                    return dict(index=task['index'],subset=task['subset'],group=group,word=word,text=text,
                        dense=base.dense_context[0].cpu().float(),tokens=tok.cpu().float(),mask=mask.cpu(),
                        geometry=geometry.cpu().float(),base=base.belief_logits[0].cpu().float(),
                        prior=torch.tensor(logits,dtype=torch.float32),region=torch.tensor(region),world=torch.tensor(world,dtype=torch.float32))
                prior,region=relation_prior(world,poly,original) if original in WORDS else (np.zeros(784),np.zeros(784,bool))
                natural.append(pack(base,inputs['tokens'][0],inputs['token_mask'][0],inputs['relation_geometry'][0],original,prior,region,task['instruction']))
                # Multi-ID binding remains a union at inference; synthetic training
                # excludes ambiguous identity and unsupported region/complement.
                if len(bound[0]['landmark']['public_ids'])!=1:continue
                try:span=replacement_span(task,names.match(task['map_name'],task['instruction']))
                except ValueError:continue
                texts=[task['instruction'][:span[0]]+w+task['instruction'][span[1]:] for w in WORDS]
                path=a.output/'current_public.jsonl';path.write_text(''.join(json.dumps(dict(records[task['index']],instruction=t))+'\n' for t in texts))
                ds=FullCandidateDataset(path,grid_size=28,max_landmarks=args['max_landmarks'],max_roads=args['max_roads'],max_relations=args['max_relations'],prior_sigma_m=args['prior_sigma_m'],rule_sigma_m=args['rule_sigma_m'])
                static,tok,mask,nodes=_batch_inputs(collate_full([ds[i] for i in range(3)]),bert,'cuda','geometry')
                for j,w in enumerate(WORDS):
                    _,region=relation_prior(world,poly,w)
                    if not region.any() or region.all():continue
                    # Relation must survive BERT truncation. Fast tokenizer provides offsets.
                    offsets=bert.tokenizer(texts[j],max_length=args['max_text_length'],truncation=True,return_offsets_mapping=True)['offset_mapping']
                    if not any(s<=span[0] and e>span[0] for s,e in offsets):continue
                    model.reset_episode();st={k:v[j:j+1] for k,v in static.items()}
                    b=model._geometry_field(st,tok[j:j+1],mask[j:j+1],nodes[j:j+1])
                    geom=relation_geometry(dict(task,anchors=bound),static['grid_xy'][j].cpu().numpy())
                    prior,_=relation_prior(world,poly,w)
                    counter.append(pack(b,tok[j],mask[j],geom,w,prior,region,texts[j]))
                if task['index']%100==0:print('cache',task['index'],len(counter),flush=True)
        atomic_save(dict(natural=natural,counter=counter,source_sha=source_sha),cachefile)
    cache_seconds=time.perf_counter()-started
    # Synthetic counterfactuals balanced per relation; no invented target coords.
    fit=[x for x in counter if x['subset']=='fit'];validation=[x for x in counter if x['subset']=='dev']
    counts={w:sum(x['word']==w for x in fit) for w in WORDS}
    labels=json.loads((a.source/'inventory_validated/private_labels.json').read_text())
    head=model.direct_target
    for module in (head.word,head.predicate,head.relative,head.relation_score):module.requires_grad_(True)
    params=[x for x in head.parameters() if x.requires_grad];opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=.001)
    def prediction(rows,train_head):
        maxn=max(len(x['tokens']) for x in rows);n=len(rows)
        tokens=torch.zeros(n,maxn,768,device='cuda');m=torch.zeros(n,maxn,dtype=torch.bool,device='cuda')
        for i,x in enumerate(rows):tokens[i,:len(x['tokens'])]=x['tokens'].cuda();m[i,:len(x['mask'])]=x['mask'].cuda()
        dense=torch.stack([x['dense'] for x in rows]).cuda();geom=torch.stack([x['geometry'] for x in rows]).cuda()
        ids=torch.tensor([[PREDICATES.index(x['word'])] for x in rows],device='cuda');valid=torch.ones_like(ids,dtype=torch.bool)
        return torch.stack([x['base'] for x in rows]).cuda()+train_head.relation(dense,tokens,m,geom,ids,valid)
    history=[];start_epoch=0;maxgrad=0.;train_start=time.perf_counter()
    if a.resume:
        saved=torch.load(a.output/'last.pt',weights_only=True)
        if saved['source_sha']!=source_sha:raise ValueError('Resume source checkpoint changed')
        head.load_state_dict(saved['head']);opt.load_state_dict(saved['optimizer']);history=saved['history'];start_epoch=saved['epoch']
    for epoch in range(start_epoch,a.epochs):
        order=np.random.default_rng(42+epoch).permutation(len(fit));losses=[]
        for offset in range(0,len(order),16):
            rows=[fit[i] for i in order[offset:offset+16]];logits=prediction(rows,head);prob=logits.softmax(-1)
            region=torch.stack([x['region'] for x in rows]).cuda();mass=(prob*region).sum(-1)
            weight=torch.tensor([len(fit)/(3*counts[x['word']]) for x in rows],device='cuda')
            uniform=region.float()/region.sum(-1,keepdim=True)
            # Balanced synthetic regional supervision; no destination labels.
            synthetic=(-(uniform*logits.log_softmax(-1)).sum(-1)+F.relu(.2-mass+(1-mass)))
            base=torch.stack([x['base'] for x in rows]).cuda();preserve=F.kl_div(logits.log_softmax(-1),base.softmax(-1),reduction='none').sum(-1)
            # Original, unmodified TRAIN queries alone receive true target labels.
            orig=[x for x in natural if x['subset']=='fit' and x['word'] in WORDS]
            rng=np.random.default_rng(42000+epoch*10000+offset);rr=[orig[i] for i in rng.integers(0,len(orig),len(rows))]
            real=prediction(rr,head);world=torch.stack([x['world'] for x in rr]).cuda();target=torch.tensor([labels[x['index']]['target_world'] for x in rr],device='cuda')
            dist=(world-target[:,None]).norm(dim=-1);soft=(-dist.square()/450.).softmax(-1)
            target_loss=-(soft*real.log_softmax(-1)).sum(-1).mean()
            loss=(weight*synthetic).mean()+target_loss+.2*preserve.mean()
            assert torch.isfinite(loss);opt.zero_grad();loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(params,5.,error_if_nonfinite=True);maxgrad=max(maxgrad,float(grad));opt.step();losses.append(float(loss.detach()))
        history.append(dict(epoch=epoch+1,loss=float(np.mean(losses))));print(history[-1],flush=True)
        atomic_save(dict(head=head.state_dict(),optimizer=opt.state_dict(),epoch=epoch+1,history=history,source_sha=source_sha),a.output/'last.pt')
    torch.cuda.synchronize();train_seconds=time.perf_counter()-train_start
    def methods(rows):
        g=torch.stack([x['base'] for x in rows]).cuda();p=torch.stack([x['prior'] for x in rows]).cuda()
        r=prediction(rows,head)
        # Fixed 50/50 mixture, not selected on confirmation labels.
        fusion=torch.logaddexp(r.log_softmax(-1),p.log_softmax(-1))-np.log(2.)
        for i,x in enumerate(rows):
            if x['word'] not in WORDS:p[i]=g[i];r[i]=g[i];fusion[i]=g[i]
        return dict(Geometry=g,Explicit=p,Relation=r,Fusion=fusion)
    results=[];cf=[];arrays={}
    with torch.no_grad():
        for x in natural:
            out=methods([x]);world=x['world'].numpy();target=np.asarray(labels[x['index']]['target_world']);d=np.linalg.norm(world-target,axis=-1)
            row={k:x[k] for k in ('index','subset','group','word','text')};row['supported']=x['word'] in WORDS;row['metrics']={}
            for method,v in out.items():
                prob=v[0].softmax(-1).cpu().numpy();idx=int(prob.argmax())
                row['metrics'][method]=dict(NE=float(d[idx]),R20=bool(d[idx]<=20),mass20=float(prob[d<=20].sum()),region_mass=float(prob[x['region'].numpy()].sum()))
                if x['subset']=='dev':arrays[f'{x["index"]}_{method}']=prob
            results.append(row)
        for x in validation:
            out=methods([x]);row={k:x[k] for k in ('index','group','word','text')};row['metrics']={}
            for method,v in out.items():
                prob=v[0].softmax(-1);region=x['region'].cuda();row['metrics'][method]=dict(mass=float(prob[region].sum()),centroid=(prob[:,None]*x['world'].cuda()).sum(0).cpu().tolist())
            cf.append(row)
    assert file_sha256(source)==source_sha
    atomic_json(results,a.output/'natural_predictions.json');atomic_json(cf,a.output/'counterfactual_predictions.json');np.savez_compressed(a.output/'beliefs.npz',**arrays)
    atomic_json(history,a.output/'history.json')
    fitgroups={x['group'] for x in fit};devgroups={x['group'] for x in validation};assert not fitgroups&devgroups
    atomic_json(dict(seed=42,epochs=a.epochs,precision='FP32, TF32 off',trainable_parameters=sum(x.numel() for x in params),
        fit_counterfactual_counts=counts,dev_counterfactual_counts={w:sum(x['word']==w for x in validation) for w in WORDS},
        fit_original_queries=sum(x['subset']=='fit' and x['word'] in WORDS for x in natural),
        natural_queries=len(natural),fit_groups=len(fitgroups),dev_groups=len(devgroups),group_overlap=0,
        cache_seconds=cache_seconds,train_seconds=train_seconds,peak_GPU_GiB=torch.cuda.max_memory_allocated()/2**30,max_gradient_norm=maxgrad,
        source_checkpoint=str(source),source_sha256=source_sha,checkpoint_sha256=file_sha256(a.output/'last.pt'),
        validation_used_for_selection=False,epochs_and_mixture_fixed_before_validation=True,
        near_radius_m=30.,region_margin=.2,preservation_KL_weight=.2,mixture=.5,geography_verified=False,
        directions_abstained=True,Val_Test_accessed=False,normal_runtime_changed=False),a.output/'manifest.json')
    print('FINISHED',flush=True)

if __name__=='__main__':main()
