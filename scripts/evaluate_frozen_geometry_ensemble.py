"""Fixed equal-probability ensemble of two existing Geometry checkpoints; TRAIN only."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from graphnav.full_experiment import new_model,_batch_inputs
from graphnav.candidate_geometry import FullCandidateDataset,collate_full
from graphnav.token_language import FrozenBERTTokens
from scripts.evaluate_distinct_ranker_soft import decode

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--stage1',type=Path,required=True);p.add_argument('--bert-cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(42)
 s=torch.load(a.stage1,map_location='cpu',weights_only=True);cfg=SimpleNamespace(**s['a3_args']);model=new_model(cfg,torch.device('cuda'));model.load_state_dict({k[3:]:v for k,v in s['model'].items() if k.startswith('a3.')});model.eval();bert=FrozenBERTTokens(cfg.text_backbone,device='cuda',max_length=96);bert.load_cache(a.bert_cache)
 ds=FullCandidateDataset(a.root/'scoped_data/public_train.jsonl',grid_size=28,max_landmarks=cfg.max_landmarks,max_roads=cfg.max_roads,max_relations=cfg.max_relations,prior_sigma_m=cfg.prior_sigma_m,rule_sigma_m=cfg.rule_sigma_m,deduplicate_queries=False);labels=json.loads((a.root/'scoped_data/labels_private.json').read_text());rows=[];a.output.mkdir(parents=True,exist_ok=True)
 with torch.no_grad():
  for l in labels:
   if l['split']!='development':continue
   i=l['index'];x,t,m,n=_batch_inputs(collate_full([ds[i]]),bert,torch.device('cuda'),'geometry');field=model(x,variant='geometry',tokens=t,token_mask=m,node_emb=n);old=field.belief_logits[0].float();z=np.load(a.cache/'public_features'/f'{i:05d}.npz');current=torch.tensor(z['base_logits_full'],device='cuda');ensemble=((current.softmax(-1)+old.softmax(-1))*.5).clamp_min(1e-30).log();pred={}
   for name,v in [('current',current),('stage1',old),('ensemble',ensemble)]:
    scores=v.cpu().numpy();pred[name]=decode(scores,z['base_world_grid'],int(scores.argmax()))
   target=np.array(l['target_world']);rows.append(dict(index=i,NE={k:float(np.linalg.norm(v-target)) for k,v in pred.items()}))
 summary={}
 for arm in ['current','stage1','ensemble']:
  summary[arm]=dict(n=len(rows),R20=float(np.mean([r['NE'][arm]<=20 for r in rows])),NE=float(np.mean([r['NE'][arm] for r in rows])),corrected=sum(r['NE']['current']>20 and r['NE'][arm]<=20 for r in rows),damaged=sum(r['NE']['current']<=20 and r['NE'][arm]>20 for r in rows))
 (a.output/'results.json').write_text(json.dumps(dict(summary=summary,rows=rows,scope='TRAIN development; fixed 50/50 probability ensemble; no fitting or Val selection'),indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
