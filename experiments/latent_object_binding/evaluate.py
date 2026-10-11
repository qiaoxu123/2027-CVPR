"""Uniform object/goal evaluation; labels enter only after public inference."""
import collections,gzip,json,time
from pathlib import Path
import numpy as np,torch
from experiments.semantic_polygon_grounding.audit import save
from experiments.bidirectional_core.stats import summarize_rows,paired_difference
from .resources import OUT,primary_features,regions
from .inference import posterior,DEFAULT_WEIGHTS

METRICS=('object_r1','object_r4','object_r8','goal_r1','goal_r4','goal_r8','ne','recovery','damage','net_recovery')


def materialize(p,mapdata,a0,device='cpu'):
    r=dict(p);r['kernels']=mapdata['kernels'].to(device);r['base']=torch.softmax(a0.float().to(device),0)
    return r


def metrics(logits,belief,mapdata,label,baseline=None):
    log=np.asarray(logits,float);world=mapdata['world'].numpy();xy=np.array([l['target_world'] for l in label['offline_labels']]);distance=np.linalg.norm(world[:,None]-xy[None],axis=-1).min(1);order=np.argsort(-log,kind='stable');positive=label['positive_objects'];bo=np.argsort(-np.asarray(belief,float),kind='stable');out={'ne':float(distance[order[0]])}
    for k in (1,4,8):out['goal_r'+str(k)]=float((distance[order[:k]]<=20).any());out['object_r'+str(k)]=float(any(i in bo[:k] for i in positive))
    base=out['goal_r1'] if baseline is None else baseline;rec=(1-base)*out['goal_r1'];damage=base*(1-out['goal_r1']);out.update(recovery=rec,damage=damage,net_recovery=rec-damage)
    return out,order,bo


def result_row(meta,label,pred,maps,a0,baseline=None):
    g=maps[meta['map_name']];log=pred['logits'].detach().float().cpu().numpy();belief=pred['object_belief'].detach().float().cpu().numpy();m,order,objects=metrics(log,belief,g,label,baseline)
    return dict(sample_id=str(meta.get('sample_id',meta['query_index'])),family_id=meta.get('family_id',meta.get('family_ids',['query:'+str(meta['query_index'])])[0]),map_name=meta['map_name'],split=meta['split'],instruction=meta['instruction'],metrics=m,goal_top8=order[:8].tolist(),object_top8=[g['ids'][i] for i in objects[:8]],object_probabilities_top8=belief[objects[:8]].tolist(),object_target_ranks=[int(np.flatnonzero(objects==i)[0]+1) for i in label['positive_objects']],offline_labels=label['offline_labels'],gate=float(pred.get('gate',0)),known_mass=float(pred.get('known_mass',0)))


def aggregate(rows):return {m:summarize_rows(rows,m,seed=20261011) for m in METRICS}
def dump(path,rows):
    with gzip.open(path,'wt') as f:
        for r in rows:f.write(json.dumps(r,allow_nan=False)+'\n')


def full_eval(weights=None,methods=('A0','A1','A2','Hard'),split='dev',device='cpu',write=True,overrides=None):
    dest=OUT/'full_geometry';maps=torch.load(dest/'maps.pt',weights_only=True);manifest=json.loads((dest/'complete.json').read_text());labels={l['query_index']:l for l in torch.load(dest/'labels.pt',weights_only=True)};rows={method:[] for method in methods};start=time.time()
    if device=='cuda':
        for g in maps.values():g['kernels']=g['kernels'].cuda()
    for chunk in manifest['chunks']:
        batch=torch.load(dest/chunk,weights_only=True,mmap=True)
        for data in batch:
            if data['split']!=split:continue
            label=labels[data['query_index']];selected=(overrides or {}).get(data['query_index'],data)
            p=materialize(selected,maps[data['map_name']],data['a0'],device);default=posterior(p,relations=True);base_metrics=metrics(data['a0'].numpy(),default['object_belief'].detach().cpu().numpy(),maps[data['map_name']],label)[0]
            for method in methods:
                if method=='A0':pred=dict(default,logits=data['a0'].to(device),gate=0)
                else:pred=posterior(p,weights if method=='LearnedGeom' else None,relations=method!='A1',hard=method=='Hard')
                row=result_row(data,label,pred,maps,data['a0'],base_metrics['goal_r1']);row['object_source']='A2 shadow, A0 has no object belief' if method=='A0' else 'latent target marginal';rows[method].append(row)
        del batch
        if len(next(iter(rows.values()))) and len(next(iter(rows.values())))%1024<64:print('full_eval',split,len(next(iter(rows.values()))),round(time.time()-start,1),flush=True)
    metrics_out={k:aggregate(v) for k,v in rows.items()};comparisons={k:{m:paired_difference(rows['A0'],v,m,seed=20261011) for m in METRICS} for k,v in rows.items() if k!='A0'} if 'A0' in rows else {}
    if write:
        for k,v in rows.items():dump(OUT/f'full_{split}_{k}.jsonl.gz',v)
        save(OUT/f'full_{split}_metrics.json',dict(metrics=metrics_out,comparisons=comparisons,seconds=time.time()-start,unit='one raw map–instruction query; multiple GT identities remain multiple positives',independent_maps=len({r['map_name'] for r in next(iter(rows.values()))}),reference_binding_truth=None))
    return rows,metrics_out


def observed_eval(models=None,write=True):
    models=models or {};maps=torch.load(OUT/'full_geometry/maps.pt',weights_only=True);allmetrics={};allcomparisons={};start=time.time()
    for split in ('train','dev'):
        feature_cache,states,feature_manifest=primary_features(split);visible={s['metadata']['sample_id']:set(regions(s)) for s in states};del feature_cache
        data=torch.load(OUT/'observed'/f'{split}_public.pt',weights_only=True,mmap=True);labels0={l['sample_id']:l for l in torch.load(OUT/'observed'/f'{split}_labels.pt',weights_only=True)};rows=collections.defaultdict(list)
        for d in data:
            meta=dict(d['metadata'],split=split,query_index=d['query_index']);g=maps[meta['map_name']];lab=labels0[meta['sample_id']];label=dict(positive_objects=[g['ids'].index(lab['target_object_id'])],offline_labels=[lab]);p=materialize(d['versions']['geometry'],g,d['a0']);base=posterior(p);bm=metrics(d['a0'].numpy(),base['object_belief'].numpy(),g,label)[0]
            for method,version,weight,hard,relations in [('A0','geometry',None,False,True),('A1','geometry',None,False,False),('A2','geometry',None,False,True),('Hard','geometry',None,True,True),('A3_HSV','hsv',None,False,True),('A3_SigLIP','siglip',None,False,True),('A3_Both','real',None,False,True),('A3_References','references_only',None,False,True),('A3_Target','target_only',None,False,True),('A5_Black','black',None,False,True),('A5_Shuffled','shuffled',None,False,True),('MatchedGeom','geometry',models.get('geometry'),False,True),('A4_Real','real',models.get('rgb'),False,True),('A4_Black','black',models.get('rgb'),False,True),('A4_Shuffled','shuffled',models.get('rgb'),False,True)]:
                if method in ('MatchedGeom','A4_Real','A4_Black','A4_Shuffled') and weight is None:continue
                pp=materialize(d['versions'][version],g,d['a0']);pred=posterior(pp,weight,hard=hard,relations=relations)
                if method=='A0':pred=dict(pred,logits=d['a0'],gate=0)
                rr=result_row(meta,label,pred,maps,d['a0'],bm['goal_r1']);rr['object_source']='A2 shadow, A0 has no object belief' if method=='A0' else 'latent target marginal';rr['reference_assignments']=d['versions'][version]['assignments'];rr['reference_posteriors']=d['versions'][version]['reference_posteriors'];rr['joint_assignment_probabilities']=pred['assignment_posterior'].detach().tolist();rr['reference_binding_marginals']={v:{i:float(p.detach()) for i,p in marginal.items()} for v,marginal in pred['reference_binding_marginals'].items()};rr['language']=d['versions'][version]['language'];rows[method].append(rr)
            tid=label['positive_objects'][0];belief=torch.zeros(len(g['ids']));belief[tid]=1;heat=g['kernels'][tid];pred=dict(logits=torch.log(heat.clamp(min=1e-30)),object_belief=belief,gate=1.,known_mass=1.);rows['GlobalIdentityOracle'].append(result_row(meta,label,pred,maps,d['a0'],bm['goal_r1']))
            # Offline perfect identity through the SAME fixed output gate. This
            # separates the public-position upper bound from interface limits.
            gate=base['gate'];mixed=(1-gate)*torch.softmax(d['a0'].float(),0)+gate*heat
            limited=dict(logits=torch.log(mixed.clamp(min=1e-30)),object_belief=belief,gate=gate,known_mass=base['known_mass']);rows['SameInterfaceGlobalIdentityOracle'].append(result_row(meta,label,limited,maps,d['a0'],bm['goal_r1']))
            # Visual support oracle MUST abstain for a target that has no legal
            # credible observed ROI. Global identity is a distinct stronger bound.
            target_observed=lab['target_object_id'] in visible[meta['sample_id']]
            observed_pred=pred if target_observed else base
            observed_limited=limited if target_observed else base
            for name,oracle in [('ObservedIdentityOracle',observed_pred),('SameInterfaceObservedIdentityOracle',observed_limited)]:
                rr=result_row(meta,label,oracle,maps,d['a0'],bm['goal_r1']);rr['oracle_target_observed']=target_observed;rows[name].append(rr)
        allmetrics[split]={k:aggregate(v) for k,v in rows.items()};pairs={'joint_minus_hard':('Hard','A2'),'geometry_joint_minus_independent':('A1','A2'),'hsv_minus_geometry':('A2','A3_HSV'),'siglip_minus_geometry':('A2','A3_SigLIP'),'rgb_minus_black':('A5_Black','A3_Both'),'rgb_minus_shuffled':('A5_Shuffled','A3_Both'),'learned_rgb_minus_geom':('MatchedGeom','A4_Real'),'learned_real_minus_black':('A4_Black','A4_Real'),'learned_real_minus_shuffled':('A4_Shuffled','A4_Real')};allcomparisons[split]={key:{m:paired_difference(rows[a],rows[b],m,seed=20261011) for m in METRICS} for key,(a,b) in pairs.items() if a in rows and b in rows}
        if write:
            for k,v in rows.items():dump(OUT/f'observed_{split}_{k}.jsonl.gz',v)
    if write:save(OUT/'observed_metrics.json',dict(metrics=allmetrics,comparisons=allcomparisons,seconds=time.time()-start,parser_oracle='not evaluated globally; controlled text-only labels do not give independent full structure truth',attribute_oracle_n=0,identity_oracle='offline only',no_seen=True,no_unseen=True))
    return allmetrics,allcomparisons

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['geometry','observed'],required=True);a=p.parse_args();torch.set_num_threads(1)
    with torch.no_grad():
        if a.stage=='geometry':
            for split in ('train','dev'):full_eval(split=split,device='cuda')
        else:observed_eval()
