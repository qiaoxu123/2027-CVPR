"""TRAIN-only frozen text intervention. Target coordinates only post-forward scoring."""
import json,argparse,copy,time,hashlib,csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from graphnav.candidate_geometry import FullCandidateDataset,collate_full
from graphnav.full_relation_training import load_full_relation_model,full_relation_inputs
from graphnav.token_language import FrozenBERTTokens
from graphnav.unified_text_ablation import forward_rule_control
from scripts.prepare_unified_v2_landmark_relation_ablation import extract_pairs,normalized_instruction
from scripts.analyze_instruction_landmark_association import PublicNames
from graphnav.structured_navigation import parse_instruction

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--catalog',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--limit',type=int);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);torch.manual_seed(42);np.random.seed(42);torch.set_num_threads(4)
 import spacy
 nlp=spacy.load('en_core_web_sm');names=PublicNames(json.load(open(a.catalog)))
 rows=[json.loads(s) for s in open(a.source) if s.strip()];rows=[r for r in rows if r['map_name'].startswith('birmingham_')];rows=rows[:a.limit] if a.limit else rows
 assert rows
 audits=[];variants={k:[] for k in ['full','landmark_relation','target_type_relation']}
 for i,r in enumerate(rows):
  text=r['instruction'];parsed=parse_instruction(text,r['map_name'],nlp,names);pairs=extract_pairs(text,names.match(r['map_name'],text));reduced=normalized_instruction(pairs,'landmark_relation');target=parsed['target'];entity=next((e for e in parsed['entities'] if e['id']==target['entity']),None)
  target_name=target.get('name') or (entity['text'] if entity else '');typ=target.get('category') or ''
  minimal='; '.join(s for s in [target_name,typ,reduced] if s)
  audits.append(dict(index=i,episode=r.get('episode_id'),map=r['map_name'],original=text,target_name=target_name,target_type=typ,reduced=reduced,minimal=minimal,pairs=pairs,ambiguity=parsed['ambiguity'],GT_used=False))
  for k,t in [('full',text),('landmark_relation',reduced),('target_type_relation',minimal)]:q=copy.deepcopy(r);q['instruction']=t;variants[k].append(q)
  if i%1000==0:print('parsed',i,flush=True)
 for k,v in variants.items():(out/f'{k}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in v))
 (out/'parsing.json').write_text(json.dumps(audits,ensure_ascii=False,indent=2))
 state=torch.load(a.checkpoint,map_location='cpu',weights_only=True);cfg=state['config']['geometry_args'];model=load_full_relation_model(state,'cuda').eval().requires_grad_(False);bert=FrozenBERTTokens(cfg['text_backbone'],device='cuda',max_length=int(cfg['max_text_length']))
 results={};ks=[1,5,10,20,50];details=[];arrays={}
 for arm in variants:
  ds=FullCandidateDataset(out/f'{arm}.jsonl',grid_size=int(cfg['grid_size']),max_landmarks=int(cfg['max_landmarks']),max_roads=int(cfg['max_roads']),max_relations=int(cfg['max_relations']),prior_sigma_m=float(cfg['prior_sigma_m']),rule_sigma_m=float(cfg['rule_sigma_m']),deduplicate_queries=False)
  accum=[];grids=[];targets=[];timings=[];start=time.time()
  for bi,batch in enumerate(DataLoader(ds,batch_size=16,shuffle=False,collate_fn=collate_full,num_workers=0)):
   with torch.no_grad():
    inp=full_relation_inputs(batch,bert,'cuda');torch.cuda.synchronize();t=time.perf_counter();res=forward_rule_control(model,inp,'full');torch.cuda.synchronize();timings.append((time.perf_counter()-t)*1000/len(batch['target_xy']));accum.append(res['relation_field'].belief_logits.float().cpu())
   grids.append(batch['grid_xy']*batch['map_sizes'][:,None]);targets.append(batch['target_xy']*batch['map_sizes'])
   if bi%100==0:print(arm,'batch',bi,flush=True)
  logits=torch.cat(accum);grid=torch.cat(grids);gt=torch.cat(targets);dist=(grid-gt[:,None]).norm(dim=-1);order=logits.argsort(dim=-1,descending=True);ordered=dist.gather(1,order);prob=logits.softmax(-1);hit={k:(ordered[:,:k].min(-1).values<=20) for k in ks};n=len(ds)
  summary=dict(N=n,**{f'R{k}_20m':float(hit[k].float().mean()) for k in ks},NE_mean=float(ordered[:,0].mean()),NE_median=float(ordered[:,0].median()),mass20=float((prob*(dist<=20)).sum(-1).mean()),grid_oracle20=float((dist.min(-1).values<=20).float().mean()),forward_ms_per_sample=float(np.mean(timings)),wall_seconds=time.time()-start)
  results[arm]=summary;arrays[arm]=dict(logits=logits.numpy(),grid=grid.numpy(),gt=gt.numpy(),r1=hit[1].numpy());np.savez_compressed(out/f'{arm}_predictions.npz',**arrays[arm])
  for i in range(n):details.append(dict(arm=arm,index=i,episode=audits[i]['episode'],map=audits[i]['map'],NE=float(ordered[i,0]),mass20=float((prob[i]*(dist[i]<=20)).sum()),**{f'hit{k}':int(hit[k][i]) for k in ks}))
  print(arm,summary,flush=True)
 base=arrays['full']['r1'];paired={k:dict(fixes=int((~base&v['r1']).sum()),damage=int((base&~v['r1']).sum())) for k,v in arrays.items()}
 report=dict(metrics=results,paired_vs_full=paired,parse_coverage=dict(target_name=sum(bool(r['target_name']) for r in audits),target_type=sum(bool(r['target_type']) for r in audits),named_relation=sum(any(x['phrase'] for x in r['pairs']) for r in audits)),checkpoint_sha=sha(a.checkpoint),source_sha=sha(a.source),catalog_sha=sha(a.catalog),seed=42,scope='Birmingham TRAIN fitting diagnostic, not independent validation; frozen text intervention, no training',checkpoint=a.checkpoint)
 (out/'summary.json').write_text(json.dumps(report,indent=2));
 with open(out/'per_task.csv','w') as f:w=csv.DictWriter(f,fieldnames=list(details[0]));w.writeheader();w.writerows(details)
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
