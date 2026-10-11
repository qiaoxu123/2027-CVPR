"""Fixed Train/Dev -> freeze -> Seen experiment, with explicitly gated stages."""
import argparse, collections, gzip, json, time, hashlib
from pathlib import Path
import numpy as np
import torch
from .objects import public_catalog,ObjectGraph
from .pipeline import forward,model_inputs,reconstruct
from .reasoning import mix
from .model import LanguageGAT,goal_loss
from experiments.semantic_polygon_grounding.audit import goal_metrics,sha,save,DATA
from experiments.bidirectional_core.stats import summarize_rows,paired_difference

OUT=Path('runs/language_object_graph')
METRICS=('goal_r1','goal_r4','goal_r8','ne','recovery','damage','net_recovery','object_r1','object_r4','object_r8')

def load(split):
    path=Path(f'runs/instance_grounding/plan_{split}/plan.pt')
    manifest=json.loads(path.with_name('manifest.json').read_text());assert sha(path)==manifest['files']['plan.pt']
    p=torch.load(path,weights_only=True,map_location='cpu');primary={}
    for s in p['states']:
        m=s['metadata'];f=m['family_id']
        if f not in primary or m['cutoff_step']>primary[f]['metadata']['cutoff_step']:primary[f]=s
    return [primary[k] for k in sorted(primary)],p['labels'],sha(path)

def row(state,label,graph,result,logits):
    m=state['metadata'];p=state['public'];metrics,order,rank=goal_metrics(logits,p['grid_xy'],p['bounds'],label['target_world'])
    original,_,_=goal_metrics(p['static_logits'],p['grid_xy'],p['bounds'],label['target_world'])
    recovery=(1-original['goal_r1'])*metrics['goal_r1'];damage=original['goal_r1']*(1-metrics['goal_r1'])
    metrics.update(recovery=float(recovery),damage=float(damage),net_recovery=float(recovery-damage))
    target=graph.byid.get(str(label['target_object_id']));object_order=np.argsort(result['object_belief'],kind='stable')[::-1]
    for k in (1,4,8):metrics['object_r'+str(k)]=float(target is not None and target in object_order[:k])
    return dict(sample_id=m['sample_id'],family_id=m['family_id'],map_name=m['map_name'],cutoff_step=m['cutoff_step'],
                instruction=m['instruction'],metrics=metrics,goal_order=order,goal_target_rank=rank,
                object_ids=[graph.ids[i] for i in object_order],object_probabilities=result['object_belief'][object_order].tolist(),
                offline_label=dict(target_object_id=label['target_object_id'],target_world=label['target_world']),
                language_graph=result['language'],relation_hypotheses=result['factors'])

def evaluate(states,labels,graphs,method,weight,*,model=None,device='cpu'):
    rows=[];elapsed=[]
    for state in states:
        graph=graphs[state['metadata']['map_name']];start=time.perf_counter()
        result=forward(graph,state['metadata']['instruction'],state['public'],method=method,weight=weight)
        if method=='A0':logits=np.asarray(state['public']['static_logits'])
        elif method=='IdentityOracle':
            # Offline identity only, normal forward does not see this value.
            tid=graph.byid.get(str(labels[state['metadata']['sample_id']]['target_object_id']))
            heat=result['kernels'][tid] if tid is not None else result['static']
            result['object_belief']=np.eye(1,len(graph.ids),tid).ravel() if tid is not None else result['object_belief']
            logits=mix(state['public']['static_logits'],heat,.5)
        elif model is not None:
            with torch.no_grad():
                p,selected=model_inputs(graph,result,device);o=model(p);full=reconstruct(result,selected,o['delta'])
            result['object_belief']=full['object_belief'].cpu().numpy();logits=mix(state['public']['static_logits'],full['heatmap'].cpu().numpy(),weight)
        else:logits=result['logits']
        elapsed.append(time.perf_counter()-start)
        r=row(state,labels[state['metadata']['sample_id']],graph,result,logits)
        r['object_metric_source']='A1 unary shadow only, not native A0' if method=='A0' else 'P5 object posterior'
        r['seconds']=elapsed[-1];rows.append(r)
    return rows,dict(seconds=sum(elapsed),mean_ms=1000*np.mean(elapsed),p95_ms=1000*np.quantile(elapsed,.95))

def aggregate(rows):return {k:summarize_rows(rows,k,seed=20261011) for k in METRICS}
def dump_rows(path,rows):
    with gzip.open(path,'wt') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def gate(rows,base):
    d=paired_difference(base,rows,'goal_r1',seed=20261011);r=sum(x['metrics']['recovery'] for x in rows);damage=sum(x['metrics']['damage'] for x in rows)
    return dict(passed=d['mean']>=.01 and d['map_cluster_95ci'][0]>0 and r>damage,delta=d,recovery=r,damage=damage)

def train_head(train,labels,graphs,*,message,device,weight):
    torch.manual_seed(20261011);model=LanguageGAT(message=message).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=.0003)
    prepared=[]
    for s in train:
        g=graphs[s['metadata']['map_name']];r=forward(g,s['metadata']['instruction'],s['public'],method='A2',weight=weight)
        p,idx=model_inputs(g,r,device);target=np.asarray(labels[s['metadata']['sample_id']]['target_world']);positive=torch.tensor(np.linalg.norm(r['world_grid']-target,axis=1)<=20,device=device)
        if positive.any():prepared.append((p,idx,r,positive))
    history=[];start=time.perf_counter();rng=np.random.default_rng(20261011)
    for epoch in range(3):
        total=0.;optimizer.zero_grad()
        for j,k in enumerate(rng.permutation(len(prepared))):
            p,idx,r,positive=prepared[k];output=model(p);full=reconstruct(r,idx,output['delta']);loss=goal_loss(full,positive);(loss/8).backward();total+=float(loss.detach())
            if (j+1)%8==0 or j+1==len(prepared):optimizer.step();optimizer.zero_grad()
        history.append(dict(epoch=epoch+1,loss=total/max(len(prepared),1)))
        print('head',message,history[-1],flush=True)
    name='gat' if message else 'matched_no_message';torch.save(dict(state_dict=model.state_dict(),config=dict(message=message,seed=20261011,epochs=3)),OUT/(name+'.pt'))
    return model,dict(history=history,seconds=time.perf_counter()-start,parameters=sum(p.numel() for p in model.parameters()),families=len(prepared),checkpoint_sha=sha(OUT/(name+'.pt')))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default=str(OUT));args=parser.parse_args()
    global_out=Path(args.output)
    if global_out!=OUT:raise ValueError('fixed output for provenance')
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'metrics.json').exists():raise RuntimeError('do not overwrite completed results')
    start=time.time();torch.set_num_threads(1);protocol=Path(__file__).with_name('protocol.json');save(OUT/'protocol.json',json.loads(protocol.read_text()))
    plans={};labels={};hashes={};bounds={}
    for split in ('train','dev','seen'):
        plans[split],labels[split],hashes[split]=load(split)
        for s in plans[split]:bounds[s['metadata']['map_name']]=s['public']['bounds']
    # Only public metadata is loaded from Seen before config freeze; no outcomes.
    raw=json.loads((DATA/'cityrefer/objects.json').read_text());catalog=public_catalog(raw);del raw
    graphs={};stats={};(OUT/'graphs').mkdir(exist_ok=True)
    for name,entries in sorted(catalog.items()):
        if name not in bounds:
            geometries=[np.asarray(o['position'])[:2] for o in entries.values()];a=np.array(geometries);bounds[name]=[*a.min(0),*a.max(0)]
        t=time.perf_counter();g=ObjectGraph(entries,bounds[name]);graphs[name]=g;payload=g.payload();path=OUT/'graphs'/f'{name}.pt';torch.save(payload,path)
        stats[name]={**payload['stats'],'seconds':time.perf_counter()-t,'sha256':sha(path),'bounds_source':'cached public grid or public object extents'}
        print('graph',name,stats[name]['nodes'],stats[name]['edges'],flush=True)
    save(OUT/'graph_audit.json',stats);save(OUT/'provenance.json',dict(protocol_sha=sha(protocol),plan_sha=hashes,populations={k:len(v) for k,v in plans.items()},public_input='all objects, descriptions removed',seed=20261011))
    metrics={};rows={};timing={}
    for split in ('train','dev'):
        metrics[split]={};rows[split]={};timing[split]={}
        for method in ('A0','A1','A2','IdentityOracle'):
            weights=[0] if method=='A0' else [.5] if method=='IdentityOracle' else [0,.25,.5]
            for weight in weights:
                key=f'{method}_w{weight:g}';r,t=evaluate(plans[split],labels[split],graphs,method,weight);rows[split][key]=r;metrics[split][key]=aggregate(r);timing[split][key]=t;dump_rows(OUT/f'{split}_{key}.jsonl.gz',r)
                print(split,key,metrics[split][key]['goal_r1']['mean'],flush=True)
    selected={}
    for method in ('A1','A2'):
        keys=[k for k in rows['dev'] if k.startswith(method+'_')]
        key=max(keys,key=lambda k:(metrics['dev'][k]['goal_r1']['mean'],-metrics['dev'][k]['damage']['mean'],-float(k.split('_w')[1])))
        selected[method]=float(key.split('_w')[1])
    geometry_gate=gate(rows['dev'][f'A2_w{selected["A2"]:g}'],rows['dev']['A0_w0'])
    stages=dict(geometry_dev_gate=geometry_gate,GAT='stopped: geometry Dev gate failed',RGB='stopped: geometry Dev gate failed',closedloop='not evaluated before localization gate',unseen='not run')
    models={};trained={};device='cuda' if torch.cuda.is_available() else 'cpu'
    if geometry_gate['passed']:
        for method,message in [('A2_matched_head',False),('A3',True)]:
            models[method],trained[method]=train_head(plans['train'],labels['train'],graphs,message=message,device=device,weight=selected['A2'])
            choices=[]
            for weight in [0,.25,.5]:
                key=f'{method}_w{weight:g}';r,t=evaluate(plans['dev'],labels['dev'],graphs,'A2',weight,model=models[method],device=device);rows['dev'][key]=r;metrics['dev'][key]=aggregate(r);timing['dev'][key]=t;dump_rows(OUT/f'dev_{key}.jsonl.gz',r);choices.append(key)
            best=max(choices,key=lambda k:(metrics['dev'][k]['goal_r1']['mean'],-metrics['dev'][k]['damage']['mean'],-float(k.split('_w')[1])));selected[method]=float(best.split('_w')[1])
        gat_delta=paired_difference(rows['dev'][f'A2_matched_head_w{selected["A2_matched_head"]:g}'],rows['dev'][f'A3_w{selected["A3"]:g}'],'goal_r1',seed=20261011)
        stages['GAT']=dict(trained=trained,matched_control_delta=gat_delta)
        stages['RGB']='not trained: requires separate supported RGB calibration; causal extraction available'
    # Freeze is written BEFORE any Seen metrics computation.
    save(OUT/'frozen_config.json',dict(selected=selected,protocol_sha=sha(protocol),geometry_dev_gate=geometry_gate,stage_status=stages,freeze_unix=time.time()))
    metrics['seen']={};rows['seen']={};timing['seen']={}
    for method in ['A0','A1','A2','IdentityOracle',*models]:
        weight=0 if method=='A0' else .5 if method=='IdentityOracle' else selected[method];key=f'{method}_w{weight:g}'
        r,t=evaluate(plans['seen'],labels['seen'],graphs,method if method not in models else 'A2',weight,model=models.get(method),device=device);rows['seen'][key]=r;metrics['seen'][key]=aggregate(r);timing['seen'][key]=t;dump_rows(OUT/f'seen_{key}.jsonl.gz',r)
        print('seen',key,metrics['seen'][key]['goal_r1']['mean'],flush=True)
    comparisons={}
    for split in rows:
        comparisons[split]={k:{metric:paired_difference(rows[split]['A0_w0'],r,metric,seed=20261011) for metric in METRICS} for k,r in rows[split].items() if k!='A0_w0'}
    seen_gate=gate(rows['seen'][f'A2_w{selected["A2"]:g}'],rows['seen']['A0_w0'])
    stages['closedloop']='stopped: Seen localization/safety gate failed' if not seen_gate['passed'] else 'eligible: existing controller evaluation required'
    save(OUT/'metrics.json',dict(metrics=metrics,comparisons=comparisons,timing=timing,selected=selected,stages=stages,seen_geometry_gate=seen_gate,seconds=time.time()-start,
         object_metrics_definition='A0 object_* is A1 unary shadow only; native A0 has no Object Belief',relation_accuracy=dict(value=None,reason='No reference-instance/relation truth; weak parsed factor consistency is not semantic accuracy'),freeze_sha=sha(OUT/'frozen_config.json')))
    print('COMPLETE',json.dumps(dict(selected=selected,stages=stages,seconds=time.time()-start)),flush=True)

if __name__=='__main__':main()
