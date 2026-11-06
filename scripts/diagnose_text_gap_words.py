"""One-word deletions/restorations on paired diagnostic examples, graph fixed."""
import argparse,json
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from graphnav.candidate_geometry import FullCandidateDataset,collate_full
from graphnav.full_relation_training import load_full_relation_model,full_relation_inputs
from graphnav.unified_text_ablation import forward_rule_control
from graphnav.token_language import FrozenBERTTokens

def main():
 import spacy
 p=argparse.ArgumentParser();p.add_argument('--previous',required=True);p.add_argument('--output',required=True);a=p.parse_args();src=Path(a.previous);out=Path(a.output);out.mkdir(exist_ok=True,parents=True);torch.set_num_threads(4);torch.manual_seed(42);audit=json.load(open(src/'parsing.json'));old=json.load(open(src/'summary.json'));base=np.load(src/'full_predictions.npz');mini=np.load(src/'target_type_relation_predictions.npz');gt=base['gt'];grid=base['grid'];fullhit=base['r1'];minhit=mini['r1'];chosen=[];used=set()
 for key,condition,limit in [('damage',fullhit&~minhit,12),('both_correct',fullhit&minhit,6),('mini_fix',~fullhit&minhit,6)]:
  counts=Counter();n=0
  for i in np.flatnonzero(condition):
   group=':'.join(audit[i]['episode'].split(':')[:2]);mp=audit[i]['map']
   if group in used or counts[mp]>=2:continue
   used.add(group);counts[mp]+=1;n+=1;chosen.append(dict(index=int(i),selection=key))
   if n>=limit:break
 state=torch.load(old['checkpoint'],map_location='cpu',weights_only=True);cfg=state['config']['geometry_args'];model=load_full_relation_model(state,'cuda').eval().requires_grad_(False);bert=FrozenBERTTokens(cfg['text_backbone'],device='cuda',max_length=int(cfg['max_text_length']));nlp=spacy.load('en_core_web_sm')
 datasets={}
 for arm in ['full','target_type_relation']:
  datasets[arm]=FullCandidateDataset(src/f'{arm}.jsonl',grid_size=int(cfg['grid_size']),max_landmarks=int(cfg['max_landmarks']),max_roads=int(cfg['max_roads']),max_relations=int(cfg['max_relations']),prior_sigma_m=float(cfg['prior_sigma_m']),rule_sigma_m=float(cfg['rule_sigma_m']),deduplicate_queries=False)
 result=[]
 for ci,case in enumerate(chosen):
  i=case['index'];r=audit[i];doc=nlp(r['original']);experiments=[]
  for t in doc:
   if not t.is_alpha and not t.like_num:continue
   experiments.append(dict(mode='drop_one_from_full',token=t.text,span=[t.idx,t.idx+len(t.text)],text=r['original'][:t.idx]+r['original'][t.idx+len(t.text):]))
  present={t.lower_ for t in nlp(r['minimal'])};seen=set()
  for t in doc:
   if t.lower_ in present or t.lower_ in seen or (not t.is_alpha and not t.like_num):continue
   seen.add(t.lower_);experiments.append(dict(mode='add_one_to_mini',token=t.text,span=[t.idx,t.idx+len(t.text)],text=r['minimal']+' ; '+t.text))
  for mode,arm in [('drop_one_from_full','full'),('add_one_to_mini','target_type_relation')]:
   group=[e for e in experiments if e['mode']==mode];item=datasets[arm][i]
   for start in range(0,len(group),16):
    part=group[start:start+16];batch=collate_full([{**item,'text':e['text']} for e in part])
    with torch.no_grad():res=forward_rule_control(model,full_relation_inputs(batch,bert,'cuda'),'full')['relation_field'].belief_logits.float().cpu().numpy()
    dist=np.linalg.norm(grid[i]-gt[i],axis=-1)
    for e,l in zip(part,res):
     order=np.argsort(-l);err=dist[order];e.update(NE=float(err[0]),R1=bool(err[0]<=20),R20=bool(err[:20].min()<=20),GT_best_rank=int(np.flatnonzero(err<=20)[0]+1),raw_token_count=len(bert.tokenizer.encode(e['text'],truncation=False)))
  def original_metrics(ar):
   order=np.argsort(-ar['logits'][i]);err=np.linalg.norm(grid[i]-gt[i],axis=-1)[order];return dict(NE=float(err[0]),R1=bool(err[0]<=20),R20=bool(err[:20].min()<=20),GT_best_rank=int(np.flatnonzero(err<=20)[0]+1))
  result.append(dict(**case,original=r['original'],minimal=r['minimal'],full_base=original_metrics(base),mini_base=original_metrics(mini),experiments=experiments));print('case',ci+1,len(chosen),'word_tests',len(experiments),flush=True)
 (out/'word_tests.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print('COMPLETE',len(result),sum(len(r['experiments']) for r in result),flush=True)
if __name__=='__main__':main()
