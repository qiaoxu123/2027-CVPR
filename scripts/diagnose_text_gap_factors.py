"""Frozen 2x2 text/graph intervention, TRAIN only. GT only used after forward."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from graphnav.candidate_geometry import FullCandidateDataset,collate_full
from graphnav.full_relation_training import load_full_relation_model,full_relation_inputs
from graphnav.unified_text_ablation import forward_rule_control
from graphnav.token_language import FrozenBERTTokens

def main():
 p=argparse.ArgumentParser();p.add_argument('--previous',required=True);p.add_argument('--output',required=True);p.add_argument('--limit',type=int);a=p.parse_args();src=Path(a.previous);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.manual_seed(42)
 old=json.load(open(src/'summary.json'));state=torch.load(old['checkpoint'],map_location='cpu',weights_only=True);cfg=state['config']['geometry_args'];model=load_full_relation_model(state,'cuda').eval().requires_grad_(False);bert=FrozenBERTTokens(cfg['text_backbone'],device='cuda',max_length=int(cfg['max_text_length']))
 datasets=[]
 for arm in ['full','target_type_relation']:
  ds=FullCandidateDataset(src/f'{arm}.jsonl',grid_size=int(cfg['grid_size']),max_landmarks=int(cfg['max_landmarks']),max_roads=int(cfg['max_roads']),max_relations=int(cfg['max_relations']),prior_sigma_m=float(cfg['prior_sigma_m']),rule_sigma_m=float(cfg['rule_sigma_m']),deduplicate_queries=False)
  if a.limit:ds.records=ds.records[:a.limit]
  datasets.append(ds)
 loaders=[DataLoader(ds,batch_size=16,shuffle=False,collate_fn=collate_full,num_workers=0) for ds in datasets]
 cache={arm:np.load(src/f'{arm}_predictions.npz') for arm in ['full','target_type_relation']};old_logits={k:v['logits'] for k,v in cache.items()};n=len(datasets[0]);acc={k:[] for k in ['mini_text_full_graph','full_text_mini_graph','full_additive_rule_off','mini_additive_rule_off']};aud=[];checks=[]
 for bi,(full,mini) in enumerate(zip(*loaders)):
  assert full['episode_id']==mini['episode_id'];assert torch.equal(full['target_xy'],mini['target_xy']);assert torch.equal(full['grid_xy'],mini['grid_xy'])
  with torch.no_grad():
   f=full_relation_inputs(full,bert,'cuda');m=full_relation_inputs(mini,bert,'cuda')
   if bi==0:
    for arm,inp in [('full',f),('target_type_relation',m)]:
     got=forward_rule_control(model,inp,'full')['relation_field'].belief_logits.cpu().numpy();error=float(np.max(np.abs(got-old_logits[arm][:len(got)])));assert error<1e-4,error;checks.append(dict(arm=arm,max_abs_error=error))
   combos={'mini_text_full_graph':({**f,'tokens':m['tokens'],'token_mask':m['token_mask']},'full'),'full_text_mini_graph':({**m,'tokens':f['tokens'],'token_mask':f['token_mask']},'full'),'full_additive_rule_off':(f,'additive_off'),'mini_additive_rule_off':(m,'additive_off')}
   for arm,(inp,mode) in combos.items():acc[arm].append(forward_rule_control(model,inp,mode)['relation_field'].belief_logits.float().cpu().numpy())
  for i in range(len(full['episode_id'])):
   row=dict(index=bi*16+i,episode=full['episode_id'][i],full_relation_count=int(full['relation_count'][i]),mini_relation_count=int(mini['relation_count'][i]))
   for key in ['nodes','node_types','node_valid','entity_valid','adjacency','edges','prior','prior_buildings','rules','relation_fields','pair_geometry']:
    aa=full[key][i];bb=mini[key][i];row[key+'_changed']=not torch.equal(aa,bb)
    if key in ['prior','rules','relation_fields']:row[key+'_max_delta']=float((aa.float()-bb.float()).abs().max())
   row['node_names_changed']=full['node_names'][i]!=mini['node_names'][i];aud.append(row)
  if bi%50==0:print('batch',bi,flush=True)
  # Resume partial batch outputs do not exist: completed arms written atomically after all rows.
 result={};base=cache['full'];dist=np.linalg.norm(base['grid'][:n]-base['gt'][:n,None],axis=-1)
 for arm,parts in acc.items():
  logits=np.concatenate(parts);assert np.isfinite(logits).all();np.savez_compressed(out/f'{arm}.npz',logits=logits);order=np.argsort(-logits,axis=-1);err=np.take_along_axis(dist,order,axis=-1);prob=np.exp(logits-logits.max(-1,keepdims=True));prob/=prob.sum(-1,keepdims=True)
  result[arm]=dict(N=n,**{f'R{k}':float((err[:,:k].min(-1)<=20).mean()) for k in [1,5,10,20,50]},NE=float(err[:,0].mean()),mass20=float((prob*(dist<=20)).sum(-1).mean()))
 out.joinpath('summary.json').write_text(json.dumps(dict(metrics=result,matched_replay_checks=checks,static_change_counts={key:sum(bool(r[key]) for r in aud) for key in aud[0] if key.endswith('_changed')},N=n),indent=2));out.joinpath('static_audit.json').write_text(json.dumps(aud,separators=(',',':')));print(result,flush=True)
if __name__=='__main__':main()
