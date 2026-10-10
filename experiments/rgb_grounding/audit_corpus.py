"""Binding/truncation/instance counts, not model input or human visual truth."""
import argparse
from collections import Counter
import json
from pathlib import Path
from transformers import AutoProcessor
from .evaluate_features import load
from experiments.bidirectional_core.cache import conservative_attributes,jsonl


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--old-data',required=True);p.add_argument('--output',required=True);args=p.parse_args();r=Path(args.runs);old=Path(args.old_data);output=Path(args.output)
    if output.exists():raise ValueError('refusing language audit overwrite')
    tokenizer=AutoProcessor.from_pretrained('google/siglip2-large-patch16-384',use_fast=False).tokenizer;result={}
    for name,feature,episodes in [('train','features_city_train',old/'train_seen/pilot_episodes.jsonl'),('dev','features_city_dev',r/'data_internal_dev/train_seen/pilot_episodes.jsonl'),('seen','features_city_seen_judged',old/'val_seen/pilot_episodes.jsonl')]:
        rows=load(r/feature);eps={x['episode_id']:x for x in jsonl(episodes)};count=Counter();kinds=Counter();primary=set();lengths={}
        for row in rows:
            m=row['metadata'];direction=m['direction'];count[direction]+=1;lengths[m['instruction']]=len(tokenizer(m['instruction'],add_special_tokens=True)['input_ids']);primary.add((m['map_name'],str(eps[m['episode_id']]['private_labels']['target_object_id'])))
            if direction!='forward':continue
            types={o['object_type'] for o,p in zip(m['roi_objects'],row['labels']['positive_roi_mask']) if p};attrs=conservative_attributes(row['phrase'],next(iter(types))) if len(types)==1 else {}
            count['bound_attribute_phrase' if attrs else 'unresolved_keep_full_instruction']+=1;kinds.update(attrs.keys())
            if any(p and any(k in o.get('attributes',{}) and o['attributes'][k]['label']!=v for k,v in attrs.items()) for o,p in zip(m['roi_objects'],row['labels']['positive_roi_mask'])):count['phrase_label_contradiction']+=1
        result[name]=dict(counts=dict(count),attribute_kinds=dict(kinds),unique_original_targets=len(primary),unique_instructions=len(lengths),full_instructions_over_64_tokens=sum(n>64 for n in lengths.values()),max_instruction_tokens=max(lengths.values()),human_verified_appearance_labels=0)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
