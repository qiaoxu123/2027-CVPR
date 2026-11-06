"""TRAIN-only paired main-Belief/predicate/angular-loss diagnosis, no Val selection."""
import argparse,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from graphnav.unified_direct_bearing_loss import DirectBearingDataset,angular_consistency_loss
from graphnav.full_relation_training import load_full_relation_model,full_relation_inputs
from graphnav.token_language import FrozenBERTTokens
from graphnav.candidate_geometry import collate_full
from graphnav.checkpointing import atomic_json,atomic_save,file_sha256
from scripts.evaluate_unified_v2_language_ablation import task_predictions


def describe(q):
 return dict(N=len(q['r1']),R20=float(q['r1'].float().mean()),R5=float(q['r5'].float().mean()),R15=float(q['r15'].float().mean()),NE=float(q['ne'].mean()),mass20=float(q['mass20'].mean()))


def paired(a,b):
 return dict(fixes=int((~a['r1']&b['r1']).sum()),damage=int((a['r1']&~b['r1']).sum()),net=int(b['r1'].sum()-a['r1'].sum()))


def main():
 p=argparse.ArgumentParser()
 for n in ('run','source','baseline','public-objects','bert-cache'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 b=torch.load(a.baseline,map_location='cpu',weights_only=True);c=b['config']['geometry_args']
 ds=DirectBearingDataset(a.source,public_objects=a.public_objects,deduplicate_queries=False,grid_size=28,max_landmarks=c['max_landmarks'],max_roads=c['max_roads'],max_relations=c['max_relations'],prior_sigma_m=c['prior_sigma_m'],rule_sigma_m=c['rule_sigma_m'])
 assert len(ds)==21878
 ids=[i for i,r in enumerate(ds.records) if r['map_name'].startswith('birmingham_')];assert len(ids)==7619
 loader=DataLoader(Subset(ds,ids),batch_size=8,shuffle=False,collate_fn=collate_full,num_workers=0)
 bert=FrozenBERTTokens(c['text_backbone'],device='cuda',max_length=c['max_text_length']);bert.load_cache(a.bert_cache)
 arms={'matched_full_text':a.baseline,**{arm:a.run/arm/'last.pt' for arm in ('predicate','angle01','angle03')}};results={};metrics={};angles={}
 with torch.no_grad():
  for name,path in arms.items():
   state=torch.load(path,map_location='cpu',weights_only=True);assert state['completed_epochs']==state['planned_epochs']==5 and not state['smoke']
   core=load_full_relation_model(state,'cuda').eval().requires_grad_(False);angle_loss_total=0.;angle_error_total=0.;angle_N=0;storage={k:[] for k in ('r1','r5','r15','ne','mass20','logits','geometry_logits')};angular=[];offset=0;started=time.perf_counter()
   for batch in loader:
    core.reset_episode();x=full_relation_inputs(batch,bert,'cuda')
    if name=='matched_full_text':x['static_inputs'].pop('anchor_predicate_ids')
    out=core.forward_full_relation(**x);z=out['field'].belief_logits.float();q=task_predictions(z,batch)
    for key,value in [('r1',q['r1']),('r5',q['r5']),('r15',q['r15']),('ne',q['top1_ne']),('mass20',q['mass20']),('logits',z.cpu()),('geometry_logits',out['geometry_field'].belief_logits.cpu())]:storage[key].append(value)
    ang=angular_consistency_loss(z,x['static_inputs']['pair_geometry'],batch['target_xy'].cuda(),x['static_inputs']['nodes'],batch['map_sizes'].cuda(),batch['angular_anchor_index'].cuda())
    angle_N+=ang['eligible_count'];angle_loss_total+=float(ang['loss'])*ang['eligible_count'];angle_error_total+=float(ang['angular_error_rad'])*ang['eligible_count']
    valid=ang['valid'];indices=torch.nonzero(valid).flatten()
    if len(indices):
     rows=torch.arange(len(z),device='cuda');safe=batch['angular_anchor_index'].cuda().clamp_min(0);rel=x['static_inputs']['pair_geometry'][rows,:,safe,:2].float();theta=torch.atan2(rel[...,1],rel[...,0]);bins=torch.arange(8,device='cuda')*math.pi/4
     kernel=(6*torch.cos(theta[:,:,None]-bins[None,None,:])).softmax(-1);belief=z.softmax(-1)*(rel.norm(dim=-1)>1/400);belief=belief/belief.sum(-1,keepdim=True).clamp_min(1e-9);prob=(belief[:,:,None]*kernel).sum(1)
     assert torch.equal(prob.argmax(-1)[valid],ang['pred_angle_bin'][valid])
     for j in indices.tolist():
      idx=ids[offset+j];gt_bin=int(ang['gt_angle_bin'][j]);pred=int(ang['pred_angle_bin'][j]);angular.append(dict(index=idx,pred_bin=pred,gt_bin=gt_bin,predicted_direction_mass=prob[j].cpu().tolist(),gt_direction_mass=float(prob[j,gt_bin]),correct=pred==gt_bin,absolute_error_deg=float(min((pred-gt_bin)%8,(gt_bin-pred)%8)*45)))
    offset+=len(batch['text'])
   results[name]={k:torch.cat(v) for k,v in storage.items()};metrics[name]=describe(results[name]);angles[name]=dict(N=len(angular),mean_angular_CE=angle_loss_total/max(angle_N,1),mean_actual_angular_error_deg=angle_error_total/max(angle_N,1)*180/math.pi,angular_8bin_accuracy=float(np.mean([x['correct'] for x in angular])) if angular else None,mean_quantized_angle_error_deg=float(np.mean([x['absolute_error_deg'] for x in angular])) if angular else None,mean_GT_direction_probability=float(np.mean([x['gt_direction_mass'] for x in angular])) if angular else None,cases=angular,seconds=time.perf_counter()-started)
   print(name,json.dumps(metrics[name]),flush=True);del core;torch.cuda.empty_cache()
 a.run.joinpath('offline').mkdir(exist_ok=True);atomic_save(dict(TRAIN_indices=ids,outputs=results),a.run/'offline/fields.pt')
 summary=dict(split='Birmingham TRAIN resubstitution, not independent Dev',metrics=metrics,angular=angles,paired={name:paired(results['predicate'],results[name]) for name in ('angle01','angle03')},predicate_vs_matched_full=paired(results['matched_full_text'],results['predicate']),source_sha256=file_sha256(a.source))
 from scripts.main_bearing_groups import strata
 from scripts.analyze_instruction_landmark_association import PublicNames
 matcher=PublicNames(ds.public);groups=[strata(ds.records[i],ds._predicates[i][1]>=0,matcher) for i in ids];group_metrics={}
 for group in groups[0]:
  mask=torch.tensor([x[group] for x in groups]);group_metrics[group]={}
  for name,q in results.items():
   group_metrics[group][name]=dict(N=int(mask.sum()),R20=float(q['r1'][mask].float().mean()) if mask.any() else None,NE=float(q['ne'][mask].mean()) if mask.any() else None)
 summary['overlapping_PUBLIC_text_strata']=group_metrics;summary['strata_are_automatic_not_human_gold']=True
 atomic_json(groups,a.run/'offline/strata.json')
 atomic_json(summary,a.run/'offline/summary.json');print(json.dumps(summary['paired'],indent=2))
if __name__=='__main__':main()
