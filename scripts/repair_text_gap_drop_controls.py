"""Keep exact untouched source characters in deletion controls; rerun only changed token IDs."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from graphnav.candidate_geometry import FullCandidateDataset,collate_full
from graphnav.full_relation_training import load_full_relation_model,full_relation_inputs
from graphnav.unified_text_ablation import forward_rule_control
from graphnav.token_language import FrozenBERTTokens

def remove_spans(text,spans):
 for start,end in sorted(spans,reverse=True):text=text[:start]+text[end:]
 return text

def main():
 p=argparse.ArgumentParser();p.add_argument('--previous',required=True);p.add_argument('--run',required=True);a=p.parse_args();src=Path(a.previous);root=Path(a.run);out=root/'drop_corrected_arrays';out.mkdir(exist_ok=True);records=json.load(open(root/'text_variants.json'));old=json.load(open(src/'summary.json'));state=torch.load(old['checkpoint'],map_location='cpu',weights_only=True);cfg=state['config']['geometry_args'];torch.set_num_threads(4);model=load_full_relation_model(state,'cuda').eval().requires_grad_(False);bert=FrozenBERTTokens(cfg['text_backbone'],device='cuda',max_length=int(cfg['max_text_length']));ds={}
 for g,arm in [('full','full'),('mini','target_type_relation')]:ds[g]=FullCandidateDataset(src/f'{arm}.jsonl',grid_size=int(cfg['grid_size']),max_landmarks=int(cfg['max_landmarks']),max_roads=int(cfg['max_roads']),max_relations=int(cfg['max_relations']),prior_sigma_m=float(cfg['prior_sigma_m']),rule_sigma_m=float(cfg['rule_sigma_m']),deduplicate_queries=False)
 counts={}
 for block in ['attributes','quantities','chain']:
  arm='full_drop_'+block;changed=[]
  for i,r in enumerate(records):
   new=remove_spans(r['original'],[tuple(x['span']) for x in r['tags'][block]]);previous=r['variants'][arm];r['variants'][arm]=new
   if bert.tokenizer.encode(new,truncation=True,max_length=int(cfg['max_text_length']))!=bert.tokenizer.encode(previous,truncation=True,max_length=int(cfg['max_text_length'])):changed.append(i)
  counts[arm]=dict(changed_tokenization_queries=len(changed),indices=changed)
  for g in ds:
   result=np.load(root/'checked_arrays'/f'{arm}@{g}.npy').copy()
   for start in range(0,len(changed),16):
    ids=changed[start:start+16];batch=collate_full([{**ds[g][i],'text':records[i]['variants'][arm]} for i in ids])
    with torch.no_grad():result[ids]=forward_rule_control(model,full_relation_inputs(batch,bert,'cuda'),'full')['relation_field'].belief_logits.float().cpu().numpy()
   np.save(out/f'{arm}@{g}.npy',result)
 (out/'text_variants.json').write_text(json.dumps(records,ensure_ascii=False,separators=(',',':')));(out/'correction_audit.json').write_text(json.dumps(counts,indent=2));print(counts)
if __name__=='__main__':main()
