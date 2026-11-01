"""Train-only diagnostic of ranking, density scale and fusion objective mismatch."""
import argparse,json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from scripts.train_object_query_belief import load,collate
from graphnav.object_query_belief import ObjectQueryRanker
from graphnav.checkpointing import atomic_json


def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--train',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 s=torch.load(a.checkpoint,map_location='cpu',weights_only=True);m=ObjectQueryRanker(relation=s['config']['relation']).cuda();m.load_state_dict(s['model']);m.eval();meta,rows=load(a.train);stats={};n=0;ranking={}
 for b in DataLoader(rows,batch_size=32,collate_fn=collate):
  x={k:(v.cuda().float() if v.is_floating_point() else v.cuda()) for k,v in b.items() if torch.is_tensor(v)}
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):q,null=m(x['features'],x['text'],x['text_mask'],x['geometry'],x['valid'])
  distance=(x['grid_world']-x['target_world'][:,None,:2]).norm(dim=-1);legal=distance<=20
  qdistance=(x['query_world']-x['target_world'][:,None,:2]).norm(dim=-1);positive=(qdistance<=20)&x['valid'];count=len(q);n+=count
  native=(x['features']@x['text'].transpose(-1,-2)).masked_fill(~x['text_mask'][:,None],-1e4).amax(-1).masked_fill(~x['valid'],-1e4)
  for name,scores in [('native',native),('trained',q.float())]:
   hit=positive.gather(1,scores.argmax(-1)[:,None])[:,0]
   ranking[name+'_positive_states']=ranking.get(name+'_positive_states',0)+int(hit.sum())
  ranking['coverage_states']=ranking.get('coverage_states',0)+int(positive.any(-1).sum())
  for sigma in [8.,10.,20.]:
   kernel=torch.exp(-torch.cdist(x['grid_world'],x['query_world']).square()/(2*sigma*sigma))*x['observed'][...,None]*x['valid'][:,None]
   for gain in [0.,.1,.3,.6]:
    with torch.no_grad():m.fusion_gain.fill_(float(torch.atanh(torch.tensor(gain))))
    updated,delta=m.update_belief(x['base_logits'],q,null,kernel,x['observed'],x['valid'])
    error=distance.gather(1,updated.argmax(-1)[:,None])[:,0]
    mass=torch.logsumexp(updated,-1)-torch.logsumexp(updated.masked_fill(~legal,-1e4),-1)
    key=f'sigma{sigma:g}_gain{gain:g}';v=stats.setdefault(key,dict(hits=0,mass_sum=0.,error_sum=0.))
    v['hits']+=int((error<=20).sum());v['mass_sum']+=float(mass.sum());v['error_sum']+=float(error.sum())
 result=dict(partition='Train only; diagnostic last checkpoint not selected deployment model',states=n,ranking=ranking,configs={k:dict(R1=v['hits']/n,mass=v['mass_sum']/n,NE=v['error_sum']/n) for k,v in stats.items()})
 atomic_json(result,Path(a.output));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
