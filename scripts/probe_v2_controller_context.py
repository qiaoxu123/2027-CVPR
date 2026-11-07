"""TRAIN-only context intervention at fixed model candidate arrival poses; no navigation."""
import json, argparse
from pathlib import Path
import torch
from graphnav.candidate_geometry import FullCandidateDataset,collate_full
from graphnav.full_experiment import _batch_inputs,new_model
from graphnav.token_language import FrozenBERTTokens
from graphnav.trajectory_policy import CandidateTrajectoryStop,calibrated_stop_probability
from types import SimpleNamespace

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stage1',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=128);a=p.parse_args()
 torch.manual_seed(42);torch.set_num_threads(4);device='cuda'
 s=torch.load(a.stage1,map_location='cpu',weights_only=True);cfg=SimpleNamespace(**s['a3_args'])
 model=new_model(cfg,torch.device(device));model.load_state_dict({k[3:]:v for k,v in s['model'].items() if k.startswith('a3.')});model.eval()
 head=CandidateTrajectoryStop(hidden=cfg.hidden).to(device).eval();head.load_state_dict({k[11:]:v for k,v in s['model'].items() if k.startswith('trajectory.')})
 bert=FrozenBERTTokens(cfg.text_backbone,device=device,max_length=96)
 ds=FullCandidateDataset(a.root/'scoped_data/public_train.jsonl',grid_size=cfg.grid_size,max_landmarks=cfg.max_landmarks,max_roads=cfg.max_roads,max_relations=cfg.max_relations,prior_sigma_m=cfg.prior_sigma_m,rule_sigma_m=cfg.rule_sigma_m,deduplicate_queries=False)
 f=torch.load(a.root/'scoped_candidate_features/features.pt',map_location='cpu',weights_only=True)
 labels=json.loads((a.root/'scoped_data/labels_private.json').read_text());indices=[r['index'] for r in labels if r['split']=='development'];indices=indices[:a.limit] if a.limit else indices;rows=[]
 with torch.no_grad():
  for i in indices:
   x,t,m,n=_batch_inputs(collate_full([ds[i]]),bert,torch.device(device),'geometry')
   with torch.autocast('cuda',dtype=torch.bfloat16): old=model(x,variant='geometry',tokens=t,token_mask=m,node_emb=n)
   xy=f['candidate_xy'][i:i+1].to(device);bounds=f['bounds'][i:i+1].to(device);world=bounds[:,:2,None].transpose(1,2)+xy*(bounds[:,None,2:]-bounds[:,None,:2]);k=xy.shape[1]
   ids=torch.cdist(xy.float(),x['grid_xy'].float()).argmin(-1)
   old_context=old.dense_context.gather(1,ids[...,None].expand(-1,-1,cfg.hidden)).half().float()
   pose=torch.cat([world[0],torch.full((k,1),30.,device=device),torch.zeros(k,1,device=device)],-1)
   probs={}
   for arm,context in [('v2',f['candidate_context'][i:i+1].to(device).float()),('stage1',old_context)]:
    inp=dict(candidate_xy=xy.expand(k,-1,-1),candidate_context=context.expand(k,-1,-1),candidate_logits=f['candidate_logits'][i:i+1].to(device).expand(k,-1),bounds=bounds.expand(k,-1),pose_world=pose,ground_level=torch.zeros(k,device=device),belief_entropy=f['belief_entropy'][i:i+1].to(device).expand(k),observed_fraction=torch.zeros(k,device=device))
    with torch.autocast('cuda',dtype=torch.bfloat16): out=head(**inp)
    probs[arm]=calibrated_stop_probability(out.stop_logits.diagonal().float(),20.).cpu().tolist()
   rows.append(dict(index=i,world_xy=world[0].cpu().tolist(),stop=probs))
 a.output.mkdir(parents=True,exist_ok=True);(a.output/'public_probe.json').write_text(json.dumps(rows))
 print('completed',len(rows),flush=True)
if __name__=='__main__':main()
