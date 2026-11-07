"""Frozen TRAIN fit/development test of public-name Polygon capacity allocation."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from graphnav.balanced_landmark_slots import balanced_landmark_slots
from graphnav.a3_runtime import public_static_record
from graphnav.public_landmark_mentions import PublicLandmarkMentions,add_mention_aliases
from graphnav.candidate_geometry import FullCandidateDataset,collate_full
from graphnav.full_experiment import _batch_inputs
from graphnav.token_language import FrozenBERTTokens
from scripts.unified_pgvb_v3_pilot import _load_base
from scripts.evaluate_distinct_ranker_soft import decode

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--trained',type=Path,required=True);p.add_argument('--bert-cache',type=Path,required=True);p.add_argument('--catalog',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
 q=[json.loads(x) for x in (a.root/'scoped_data/public_train.jsonl').read_text().splitlines()];labels=json.loads((a.root/'scoped_data/labels_private.json').read_text());objects=json.loads(a.catalog.read_text());matcher=PublicLandmarkMentions(objects);catalog=SimpleNamespace(objects=objects);changed=[];indices=[]
 for l in labels:
  if l['split'] not in ['fit','development']:continue
  i=l['index'];old=q[i]
  if len(old['landmarks'])<16:continue
  mentions=matcher.match(old['map_name'],old['instruction']);refs=tuple(dict.fromkeys(n for m in mentions for n in m['public_names']));b=old['world_bounds'];transform=SimpleNamespace(x_min=b[0],y_min=b[1],x_max=b[2],y_max=b[3]);ep=SimpleNamespace(map_name=old['map_name'],instruction=old['instruction'],reference_names=refs,references=());new=public_static_record(ep,transform,catalog,max_landmarks=10000);new['landmarks']=balanced_landmark_slots(new['landmarks']);new=add_mention_aliases(new,mentions)
  if new['landmarks']!=old['landmarks']:changed.append(new);indices.append(i)
 path=a.output/'public_queries.jsonl';path.write_text(''.join(json.dumps(x)+'\n' for x in changed));model,cfg=_load_base(a.source,'cuda');s=torch.load(a.trained,map_location='cpu',weights_only=True);model.load_state_dict({k[9:]:v for k,v in s['model'].items() if k.startswith('geometry.')});model.eval();bert=FrozenBERTTokens(cfg.text_backbone,device='cuda',max_length=96);bert.load_cache(a.bert_cache);ds=FullCandidateDataset(path,grid_size=28,max_landmarks=16,max_roads=cfg.max_roads,max_relations=cfg.max_relations,prior_sigma_m=cfg.prior_sigma_m,rule_sigma_m=cfg.rule_sigma_m,deduplicate_queries=False);rows=[]
 with torch.no_grad():
  for j,i in enumerate(indices):
   x,t,m,n=_batch_inputs(collate_full([ds[j]]),bert,torch.device('cuda'),'geometry');field=model(x,variant='geometry',tokens=t,token_mask=m,node_emb=n);v=field.belief_logits[0].float().cpu().numpy();z=np.load(a.cache/'public_features'/f'{i:05d}.npz');xy=z['base_world_grid'];old=decode(z['base_logits_full'],xy,int(z['base_logits_full'].argmax()));new=decode(v,xy,int(v.argmax()));target=np.array(labels[i]['target_world']);rows.append(dict(index=i,split=labels[i]['split'],before_NE=float(np.linalg.norm(old-target)),after_NE=float(np.linalg.norm(new-target))))
 summary={}
 for split in ['fit','development']:
  rr=[r for r in rows if r['split']==split];summary[split]=dict(affected=len(rr),corrected=sum(r['before_NE']>20 and r['after_NE']<=20 for r in rr),damaged=sum(r['before_NE']<=20 and r['after_NE']>20 for r in rr),before_NE=float(np.mean([r['before_NE'] for r in rr])),after_NE=float(np.mean([r['after_NE'] for r in rr])))
 (a.output/'results.json').write_text(json.dumps(dict(summary=summary,rows=rows),indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
