"""Fixed identity evaluation; targets enter only metric/Oracle functions."""
import argparse,json,time,math
from pathlib import Path
import numpy as np,torch
from experiments.rgb_grounding.head import VisualEvidenceNetwork
from experiments.rgb_grounding.train_head import collate as old_collate
from experiments.bidirectional_core.stats import summarize_rows,paired_difference
from experiments.visual_goal_association.association import associate,identity_oracle
from graphnav.memory import ObservedRGBMemory
from graphnav.checkpointing import file_sha256
from .dataset import city_rows,external_rows,collate
from .model import InstanceVEN
from .train import score
from .data import write
from .association import weights as public_weights

SEED=20261011

def load_model(checkpoint):
    if checkpoint=='native':return None
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if 'instance_model' in state:model=InstanceVEN(relative_attention=state.get('relative_attention',False));model.load_state_dict(state['instance_model'],strict=True)
    else:model=VisualEvidenceNetwork(dimension=state['dimension'],token_attention=state['token_attention']);model.load_state_dict(state['model'],strict=True)
    return model.cuda().eval().requires_grad_(False)

@torch.inference_mode()
def predict_scores(model,rows,control):
    if isinstance(model,InstanceVEN):return score(model,rows,control=control)
    predictions=[]
    for r in rows:
        p,_,_=collate([r],control);p={k:v.cuda() for k,v in p.items()}
        if model is None:
            s=(torch.nn.functional.normalize(p['images'],dim=-1)*torch.nn.functional.normalize(p['full'],dim=-1)[:,None]).sum(-1)*20
        else:
            q=dict(text=p['full'],images=p['images'],patch_tokens=p['patch_tokens'],language_tokens=p['language_tokens'],language_valid=p['language_valid'],quality=p['quality'],valid=p['valid']);s=model(q).score
        n=len(r['public']['roi_valid']);prob=s.softmax(-1)
        tri=torch.stack([torch.log(prob.clamp_min(1e-8)),torch.log((1-prob).clamp_min(1e-8)),torch.full_like(prob,-10)],-1)
        predictions.append(dict(score=s[0,:n].cpu(),prob=tri.softmax(-1)[0,:n].cpu(),tri=tri[0,:n].cpu()))
    return predictions

def family_rows(rows,key):
    grouped={}
    for row in rows:
        value=row['metrics'].get(key)
        if value is None:continue
        f=row['family_id'];g=grouped.setdefault(f,dict(map=row['map_name'],values=[]));assert g['map']==row['map_name'];g['values'].append(value)
    return [dict(sample_id=f,family_id=f,map_name=v['map'],metrics={key:float(np.mean(v['values']))}) for f,v in sorted(grouped.items())]

def summary(rows):
    result={}
    for k in rows[0]['metrics']:
        result[k]=summarize_rows(rows,k,seed=SEED)
        result[k]['family_equal_weight']=summarize_rows(family_rows(rows,k),k,seed=SEED)
    return result

def wilson(successes,n):
    if n==0:return 0
    p=successes/n;z=1.959963984540054
    return (p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)

def calibrated_match(pred,temperature):
    return (pred['tri']/temperature).softmax(-1)[:,0]

def calibration(rows,preds):
    # Temperature on observed training labels; ambiguous/absent are Unknown.
    values=[]
    for t in [.5,1,2,4]:
        losses=[]
        for r,p in zip(rows,preds):
            lab=r['labels'];n=len(r['public']['roi_valid'])
            if not n:continue
            target=torch.full((n,),2,dtype=torch.long);target[lab['semantic_valid_mask']&~lab['positive_roi_mask']]=1;target[lab['positive_roi_mask']]=0
            losses.append(float(torch.nn.functional.cross_entropy(p['tri']/t,target)))
        values.append((np.mean(losses),t))
    t=min(values)[1];chosen=1.01;audits=[]
    for threshold in [.8,.9,.95,1.01]:
        accepted={}
        for r,p in zip(rows,preds):
            if not len(p['score']):continue
            best=int(p['score'].argmax());conf=float(calibrated_match(p,t)[best]);credible=r['metadata']['roi_objects'][best]['depth_support']>=.9
            if conf>=threshold and credible:
                accepted.setdefault(r['metadata']['family_id'],[]).append(float(r['labels']['identity_positive'][best]))
        successes=sum(all(x==1 for x in v) for v in accepted.values());n=len(accepted);lower=wilson(successes,n)
        audits.append(dict(threshold=threshold,independent_families=n,all_accepted_prefixes_correct_families=successes,wilson_95_lower=lower))
        if chosen==1.01 and n>=10 and lower>=.9:chosen=threshold
    return dict(temperature=t,threshold=chosen,gain=.025,max_abs_residual=.025,calibration_audit=audits,temperature_nll=[dict(t=t,nll=float(v)) for v,t in values],confidence_note='weak original identity correctness; not human calibrated semantic truth')

def public_fusion(state,regions,pred,config):
    # No target IDs/coordinates here. Instance -> actual visited footprint ->
    # unchanged centroid grid. Abstained and unobserved cells remain identical.
    if config['gain']!=.025:raise ValueError('P2 fixed residual gain; amplification forbidden')
    logits=np.array(state['public']['static_logits'],float)
    weights=state.get('_association')
    if weights is None:weights=public_weights(state,regions)
    residual=np.zeros_like(logits);accepted=None
    if len(regions):
        best=int(pred['score'].argmax());conf=float(calibrated_match(pred,config['temperature'])[best])
        if conf>=config['threshold'] and regions[best]['depth_support']>=.9:
            accepted=best;residual=weights[best]*config['gain']*min(1,max(0,conf))
    assert abs(residual).max()<=.025+1e-12
    return logits+residual,weights,residual,accepted

def rows_metrics(rows,predictions,config,oracle=False,geometry=False):
    records=[]
    for r,p in zip(rows,predictions):
        m=r['metadata'];lab=r['labels'];state=r['state'];n=len(p['score']);regions=m['roi_objects']
        updated,w,residual,accepted=public_fusion(state,regions,p,config)
        if geometry:updated=np.array(state['public']['static_logits']);residual*=0;accepted=None
        if oracle:
            updated,known,residual=identity_oracle(state['public']['static_logits'],w,lab['identity_positive'].tolist());accepted=None
        bounds=np.array(state['public']['bounds']);xy=bounds[:2]+np.array(state['public']['grid_xy'])*(bounds[2:]-bounds[:2]);dist=np.linalg.norm(xy-np.array(r['sealed']['target_world']),axis=1)
        before=np.argsort(np.array(state['public']['static_logits']),kind='stable')[::-1];after=np.argsort(updated,kind='stable')[::-1];a=dist[before[0]]<=20;b=dist[after[0]]<=20
        pos=lab['identity_positive'];identity_scores=p['score'].clone()
        if oracle and pos.any():identity_scores[pos]+=float(identity_scores.max()-identity_scores.min()+1)
        rank=identity_scores.argsort(descending=True).tolist();visible=bool(pos.any());eligible=visible and n>=2
        target_idx=int(pos.nonzero().flatten()[0]) if visible else None
        def masked_match(mask,positive):
            ids=mask.nonzero().flatten().tolist()
            if len(ids)<2 or not (positive&mask).any():return None
            best=max(ids,key=lambda j:float(identity_scores[j]));return float(positive[best])
        metric=dict(goal_r1=float(b),goal_r4=float((dist[after[:4]]<=20).any()),goal_r8=float((dist[after[:8]]<=20).any()),ne=float(dist[after[0]]),recovery=float(not a and b),damage=float(a and not b),net_recovery=float(b)-float(a),original_instance_recovery=float(not a and b and ((accepted is not None and bool(pos[accepted])) or (oracle and visible))),incidental_recovery=float(not a and b and not ((accepted is not None and bool(pos[accepted])) or (oracle and visible))),identity_r1=float(pos[rank[0]]) if eligible else None,identity_r4=float(any(pos[j] for j in rank[:4])) if eligible else None,same_class_r1=masked_match(lab['same_class'],pos),same_color_r1=masked_match(lab['same_color'],pos),attribute_r1=masked_match(lab['attribute_judged'],lab['attribute_positive']),credible_error=float(not a and visible and lab['target_credible']),target_visible=float(visible),target_credible=float(visible and lab['target_credible']),accepted=float(accepted is not None),identity_high_pixels=float(pos[rank[0]]) if eligible and regions[target_idx]['native_min_side']>=32 else None,identity_low_pixels=float(pos[rank[0]]) if eligible and regions[target_idx]['native_min_side']<32 else None)
        if oracle:metric['attribute_r1']=None
        if geometry:
            for key in ['identity_r1','identity_r4','same_class_r1','same_color_r1','attribute_r1','identity_high_pixels','identity_low_pixels']:metric[key]=None
        records.append(dict(sample_id=m['sample_id'],family_id=m['family_id'],map_name=m['map_name'],cutoff_step=m['cutoff_step'],metrics=metric,instruction=m['instruction'],target_object_id=r['sealed']['target_object_id'],candidate_object_ids=[v['object_id'] for v in regions],candidate_source_steps=[v['source_step'] for v in regions],scores=identity_scores.tolist(),oracle_only=oracle,geometry_only=geometry,match_prob=calibrated_match(p,config['temperature']).tolist(),judgment_prob=(p['tri']/config['temperature']).softmax(-1).tolist(),static_top1=int(before[0]),visual_top1=int(after[0]),accepted_candidate=accepted,max_abs_residual=float(abs(residual).max()),target_rank=next((i+1 for i,j in enumerate(rank) if pos[j]),None),annotation_status='weak original-target identity and language attributes, human audit 0'))
    return records

def bind_contours(rows):
    from experiments.bidirectional_core.cache import DEFAULT_DATA
    objects=json.loads((Path(DEFAULT_DATA)/'cityrefer/objects.json').read_text())
    for r in rows:
        for v in r['metadata']['roi_objects']:v['contour']=objects[r['metadata']['map_name']][v['object_id']]['contour']
        r['state']['_association']=public_weights(r['state'],r['metadata']['roi_objects'])

def calibrate(a):
    out=Path(a.output)
    if out.exists():raise ValueError('no overwrite')
    out.mkdir(parents=True);torch.set_num_threads(2);rows=city_rows(a.features);bind_contours(rows);heads=json.loads(Path(a.heads).read_text());configs={};scores={};allstats={}
    for name,path in heads.items():
        model=load_model(path);pred=predict_scores(model,rows,'real');config=calibration(rows,pred);configs[name]=config;records=rows_metrics(rows,pred,config);allstats[name]=summary(records)
        scores[name]=allstats[name]['identity_r1']['family_equal_weight']['mean'];torch.save(pred,out/(name+'_scores.pt'))
        (out/(name+'_predictions.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in records));del model;torch.cuda.empty_cache()
    candidates=[n for n in heads if n.endswith('_city') or n=='city_only'];selected=max(candidates,key=lambda n:(scores[n] if scores[n] is not None else -1,n))
    write(out/'frozen_config.json',dict(complete=True,heads=heads,configs=configs,selected=selected,selection='internal Dev family-balanced original identity R1; no Seen results read',checkpoint_sha256={n:file_sha256(p) for n,p in heads.items() if p!='native'},protocol_sha256=file_sha256(Path(__file__).with_name('protocol.json')),source_sha256={p.name:file_sha256(p) for p in Path(__file__).parent.glob('*.py')},dev_feature_sha256=file_sha256(Path(a.features)/'manifest.json')))
    write(out/'metrics.json',dict(models=allstats,selection_scores=scores,selected=selected));print(json.dumps(dict(selected=selected,scores=scores,configs=configs)),flush=True)

def evaluate(a):
    out=Path(a.output)
    if out.exists():raise ValueError('no overwrite')
    out.mkdir(parents=True);torch.set_num_threads(2);start=time.time();frozen=json.loads(Path(a.config).read_text())
    for name,path in frozen['heads'].items():
        if path!='native' and file_sha256(path)!=frozen['checkpoint_sha256'][name]:raise ValueError('checkpoint changed after freeze')
    for name,sha in frozen['source_sha256'].items():
        if name in ['data.py','dataset.py','model.py','evaluate.py','train.py','relations.py','association.py'] and file_sha256(Path(__file__).with_name(name))!=sha:raise ValueError('experimental source changed after freeze')
    if file_sha256(Path(__file__).with_name('protocol.json'))!=frozen['protocol_sha256']:raise ValueError('protocol changed after freeze')
    rows=city_rows(a.features);bind_contours(rows);results={}
    for name,path in frozen['heads'].items():
        model=load_model(path);config=frozen['configs'][name]
        for control in ['real','zero','shuffled']:
            pred=predict_scores(model,rows,control);results[name+'_'+control]=rows_metrics(rows,pred,config)
            if name=='native' and control=='real':
                results['geometry_only']=rows_metrics(rows,pred,config,geometry=True);results['identity_oracle']=rows_metrics(rows,pred,config,oracle=True)
        del model;torch.cuda.empty_cache()
    for name,recs in results.items():(out/(name+'_predictions.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in recs))
    summaries={n:summary(r) for n,r in results.items()};selected=frozen['selected'];real=results[selected+'_real'];base=results['geometry_only'];paired={}
    for control,other in [('native',results['native_real']),('zero',results[selected+'_zero']),('shuffled',results[selected+'_shuffled']),('geometry',base)]:
        paired[control]={k:paired_difference(other,real,k,seed=SEED) for k in (['goal_r1','net_recovery'] if control=='geometry' else ['goal_r1','identity_r1','net_recovery'])}
        if control!='geometry':paired[control]['identity_family_equal']=paired_difference(family_rows(other,'identity_r1'),family_rows(real,'identity_r1'),'identity_r1',seed=SEED)
    credible=[r for r in real if r['metrics']['credible_error']];families=len({r['family_id'] for r in credible});p=paired
    def lower(v):return v['map_cluster_95ci'][0] if v['map_cluster_95ci'] else -1
    gates=dict(credible_independent_error_families=families>=50,instance_improvement=(p['native']['identity_family_equal']['mean'] or 0)>=.1 and lower(p['native']['identity_family_equal'])>0,real_zero=lower(p['zero']['identity_family_equal'])>0 and lower(p['zero']['goal_r1'])>0,real_shuffled=lower(p['shuffled']['identity_family_equal'])>0 and lower(p['shuffled']['goal_r1'])>0,safe_net_recovery=sum(r['metrics']['net_recovery'] for r in real)>0,goal_improvement=(p['geometry']['goal_r1']['mean'] or 0)>=.01 and lower(p['geometry']['goal_r1'])>0)
    write(out/'metrics.json',dict(models=summaries,paired_selected=paired,selected=selected,gates=gates,all_pass=all(gates.values()),credible_error_states=len(credible),credible_error_families=families,credible_error_objects=len({(r['map_name'],r['target_object_id']) for r in credible}),human_verified_semantic_cases=0,seconds=time.time()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),seed=SEED,config_sha256=file_sha256(a.config),feature_manifest_sha256=file_sha256(Path(a.features)/'manifest.json'),no_navigation_rollout=not all(gates.values()),note='Weak identity annotation; geometric/depth credible is not human semantic credibility; no future or GT input to normal scores'))
    print(json.dumps(dict(selected=selected,gates=gates,independent_error_families=families,metrics={n:{k:v[k]['mean'] for k in ['goal_r1','identity_r1','recovery','damage']} for n,v in summaries.items()})),flush=True)

def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='mode',required=True);q=s.add_parser('calibrate');q.add_argument('--features',required=True);q.add_argument('--heads',required=True);q.add_argument('--output',required=True)
    q=s.add_parser('evaluate');q.add_argument('--features',required=True);q.add_argument('--config',required=True);q.add_argument('--output',required=True);a=p.parse_args();(calibrate if a.mode=='calibrate' else evaluate)(a)
if __name__=='__main__':main()
