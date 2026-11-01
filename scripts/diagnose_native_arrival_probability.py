"""Train-only calibration, Dev verification of native visual arrival confirmation."""
import argparse,json,sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from scripts.train_object_query_belief import load,collate
from graphnav.object_query_belief import ObjectQueryRanker
from graphnav.workspace_paths import PROJECT,WORKSPACE
from graphnav.checkpointing import atomic_json


def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--train',required=True);p.add_argument('--dev',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 sys.path.insert(0,str(PROJECT/'Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source'))
 from sbfnav.dataset import CityNavDataset
 source=CityNavDataset(WORKSPACE/'DATA/sbfnav-data','train_seen',version='processed_citynav')
 state=torch.load(a.checkpoint,map_location='cpu',weights_only=True);m=ObjectQueryRanker(relation=state['config']['relation']).cuda();m.load_state_dict(state['model']);m.eval().requires_grad_(False)
 results={}
 for partition,directory in [('train',a.train),('dev',a.dev)]:
  _,rows=load(directory);truth=[];predictions={r:[] for r in [5.,10.,15.,20.]}
  for b in DataLoader(rows,batch_size=32,collate_fn=collate):
   x={k:(v.cuda().float() if v.is_floating_point() else v.cuda()) for k,v in b.items() if torch.is_tensor(v)}
   # Agent pose is public; target is accessed only below for calibration scoring.
   pose=torch.tensor([source[e].trajectory[t][:2].tolist() for e,t in zip(b['episode'],b['step'])],device='cuda')
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):q,null=m(x['features'],x['text'],x['text_mask'],x['geometry'],x['valid'])
   probability=torch.cat((q.float(),null.float()[:,None]),-1).softmax(-1)[:,:-1]
   distance=(x['query_world']-pose[:,None]).norm(dim=-1)
   for radius in predictions:predictions[radius].extend((probability*(distance<=radius)*x['valid']).sum(-1).cpu().tolist())
   truth.extend(((pose-x['target_world'][:,:2]).norm(dim=-1)<=20).cpu().tolist())
  target=torch.tensor(truth);configs={}
  for radius,values in predictions.items():
   probability=torch.tensor(values)
   for threshold in [.3,.5,.7,.8,.9]:
    confirmed=probability>=threshold;tp=int((confirmed&target).sum());fp=int((confirmed&~target).sum());n=int(confirmed.sum())
    configs[f'{radius:g}/{threshold:g}']=dict(radius=radius,threshold=threshold,confirmed=n,true_positive=tp,false_positive=fp,precision=tp/n if n else None,recall=tp/int(target.sum()) if target.any() else None)
  results[partition]=dict(states=len(rows),positive_arrivals=int(target.sum()),configs=configs)
 eligible=[v for v in results['train']['configs'].values() if v['confirmed']>=20 and v['precision']>=.95]
 chosen=max(eligible,key=lambda v:(v['recall'],v['precision'],-v['radius'])) if eligible else None
 selected_dev=results['dev']['configs'][f"{chosen['radius']:g}/{chosen['threshold']:g}"] if chosen else None
 atomic_json(dict(results=results,train_calibrated_rule=chosen,dev_verification=selected_dev,protocol='Query posterior mass within public agent radius. Target coordinates score calibration only. Select on Train with precision>=95% and>=20 confirmations, verify once on prepared Dev; no navigation success claim.'),Path(a.output))
 print(json.dumps(dict(train_positives=results['train']['positive_arrivals'],dev_positives=results['dev']['positive_arrivals'],train_calibrated_rule=chosen,dev_verification=selected_dev),indent=2))


if __name__=='__main__':main()
