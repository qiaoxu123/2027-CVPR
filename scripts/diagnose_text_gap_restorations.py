"""Paired per-task text restorations with graph held fixed, no labels in inference."""
import argparse,json,re,time
from pathlib import Path
from collections import deque
import numpy as np
import torch
from torch.utils.data import DataLoader
from graphnav.candidate_geometry import FullCandidateDataset,collate_full
from graphnav.full_relation_training import load_full_relation_model,full_relation_inputs
from graphnav.unified_text_ablation import forward_rule_control
from graphnav.token_language import FrozenBERTTokens
from graphnav.structured_navigation import parse_instruction
from scripts.analyze_instruction_landmark_association import PublicNames

QUANTITY={'first','second','third','fourth','fifth','sixth','seventh','eighth','ninth','tenth','last','nearest','closest','farthest','furthest','row','rows','column','columns','meter','meters','metre','metres','feet','foot','yard','yards','leftmost','rightmost','left-most','right-most'}

def build_variants(text,map_name,nlp,names,minimal):
 doc=nlp(text);parsed=parse_instruction(text,map_name,lambda _:doc,names);target=parsed['target']['entity'];ti=int(target[1:]) if target else None
 def intersects(t,span):return t.idx<span[1] and t.idx+len(t.text)>span[0]
 attr={t.i for t in doc for a in parsed['target']['attributes'] if intersects(t,a['span'])}
 quant={t.i for t in doc if t.like_num or t.lemma_.lower() in QUANTITY or re.fullmatch(r'\d+(?:st|nd|rd|th)',t.lower_)}
 attr-=quant  # Ordinals/numbers belong to the quantity intervention, not appearance.
 # Dependency spine links target to each public mention; no world labels involved.
 chain=set();adj={t.i:[] for t in doc}
 for t in doc:
  if t.head.i!=t.i:adj[t.i].append(t.head.i);adj[t.head.i].append(t.i)
 if ti is not None:
  for mention in parsed['landmarks']:
   anchors=[t for t in doc if intersects(t,[mention['start'],mention['end']])]
   if not anchors:continue
   ids={t.i for t in anchors};anchor=next((t for t in anchors if t.head.i not in ids),anchors[0]);queue=deque([(ti,[ti])]);seen={ti}
   while queue:
    u,path=queue.popleft()
    if u==anchor.i:chain.update(path);break
    for v in adj[u]:
     if v not in seen:seen.add(v);queue.append((v,path+[v]))
 named={t.i for t in doc for m in parsed['landmarks'] if intersects(t,[m['start'],m['end']])}
 # Preserve local object modifiers for intermediate NPs, but do not leak target attrs into chain arm.
 middle={t.i for t in doc if t.i in chain and t.pos_ in ('NOUN','PROPN') and t.i!=ti and t.i not in named}
 for chunk in doc.noun_chunks:
  if chunk.root.i in middle:chain.update(t.i for t in chunk)
 chain-=named|({ti} if ti is not None else set());chain-=attr|quant
 def words(ids):return ' '.join(t.text for t in doc if t.i in ids)
 def drop(ids):return ' '.join(t.text for t in doc if t.i not in ids)
 def add(ids):return minimal+(' ; '+words(ids) if ids else '')
 # Restore exact source tokens, including duplicates where already present; no invented facts.
 variants={'restore_attributes':add(attr),'restore_quantities':add(quant),'restore_chain':add(chain),'restore_all':add(attr|quant|chain),'reorder_minimal':'; '.join(reversed(minimal.split('; '))),'full_drop_attributes':drop(attr),'full_drop_quantities':drop(quant),'full_drop_chain':drop(chain)}
 return variants,dict(attributes=[dict(text=t.text,span=[t.idx,t.idx+len(t.text)]) for t in doc if t.i in attr],quantities=[dict(text=t.text,span=[t.idx,t.idx+len(t.text)]) for t in doc if t.i in quant],chain=[dict(text=t.text,span=[t.idx,t.idx+len(t.text)]) for t in doc if t.i in chain],intermediate_nouns=[doc[i].text for i in sorted(middle)],indirect_proxy=bool(parsed['indirect_relations']),direct_proxy=bool(parsed['direct_relations']),ambiguity=parsed['ambiguity'])

def main():
 import spacy
 p=argparse.ArgumentParser();p.add_argument('--previous',required=True);p.add_argument('--catalog',required=True);p.add_argument('--output',required=True);p.add_argument('--limit',type=int);a=p.parse_args();src=Path(a.previous);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.manual_seed(42)
 old=json.load(open(src/'summary.json'));audits=json.load(open(src/'parsing.json'));audits=audits[:a.limit] if a.limit else audits;nlp=spacy.load('en_core_web_sm');names=PublicNames(json.load(open(a.catalog)));prepared=out/'text_variants.json'
 if prepared.exists():records=json.load(open(prepared));assert len(records)==len(audits)
 else:
  records=[]
  for i,r in enumerate(audits):
   variants,tags=build_variants(r['original'],r['map'],nlp,names,r['minimal']);records.append(dict(index=i,original=r['original'],minimal=r['minimal'],variants=variants,tags=tags))
   if i%1000==0:print('prepare',i,flush=True)
  prepared.write_text(json.dumps(records,ensure_ascii=False,separators=(',',':')))
 state=torch.load(old['checkpoint'],map_location='cpu',weights_only=True);cfg=state['config']['geometry_args'];model=load_full_relation_model(state,'cuda').eval().requires_grad_(False);bert=FrozenBERTTokens(cfg['text_backbone'],device='cuda',max_length=int(cfg['max_text_length']))
 datasets=[]
 for arm in ['full','target_type_relation']:
  ds=FullCandidateDataset(src/f'{arm}.jsonl',grid_size=int(cfg['grid_size']),max_landmarks=int(cfg['max_landmarks']),max_roads=int(cfg['max_roads']),max_relations=int(cfg['max_relations']),prior_sigma_m=float(cfg['prior_sigma_m']),rule_sigma_m=float(cfg['rule_sigma_m']),deduplicate_queries=False)
  if a.limit:ds.records=ds.records[:a.limit]
  datasets.append(ds)
 arms=list(records[0]['variants']);n=len(records);acc={f'{k}@{g}':np.zeros((n,784),np.float32) for k in arms for g in ['mini','full']};token_lengths={k:[] for k in arms};start=time.time();loader=[DataLoader(ds,batch_size=16,shuffle=False,collate_fn=collate_full,num_workers=0) for ds in datasets];bdir=out/'batches';bdir.mkdir(exist_ok=True)
 for bi,(bf,bm) in enumerate(zip(*loader)):
  offset=bi*16;end=min(offset+16,n);checkpoint=bdir/f'{bi:04d}.npz'
  if checkpoint.exists():
   saved=np.load(checkpoint)
   for key in acc:acc[key][offset:end]=saved[key]
   continue
  with torch.no_grad():
   inputs={'full':full_relation_inputs(bf,bert,'cuda'),'mini':full_relation_inputs(bm,bert,'cuda')};step={}
   for arm in arms:
    texts=[r['variants'][arm] for r in records[offset:end]];tokens,mask=bert.encode_tokens(texts);token_lengths[arm]+=mask.sum(-1).cpu().tolist()
    for g,inp in inputs.items():
     override={**inp,'tokens':tokens.float(),'token_mask':mask};logits=forward_rule_control(model,override,'full')['relation_field'].belief_logits.float().cpu().numpy();assert np.isfinite(logits).all();acc[f'{arm}@{g}'][offset:end]=logits;step[f'{arm}@{g}']=logits
   np.savez_compressed(checkpoint,**step)
  if bi%25==0:print('batch',bi,'/',(n+15)//16,'elapsed',round(time.time()-start),flush=True)
 for key,v in acc.items():np.savez_compressed(out/f'{key}.npz',logits=v)
 out.joinpath('run.json').write_text(json.dumps(dict(N=n,arms=list(acc),wall_seconds=time.time()-start,token_lengths=token_lengths,graph_definition='mini/full original public-text derived tensors held fixed; only BERT tokens/mask changed; no GT inputs'),indent=2));print('COMPLETE',flush=True)
if __name__=='__main__':main()
