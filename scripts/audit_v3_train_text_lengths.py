"""TRAIN-only tokenizer audit; no GT labels, model forward or parameter tuning."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoConfig,AutoTokenizer
from graphnav.checkpointing import atomic_json,file_sha256

def main():
    p=argparse.ArgumentParser()
    for k in ['source-v2','train-jsonl','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();cfg=torch.load(a.source_v2,map_location='cpu',weights_only=True)['config']['geometry_args']
    # Read only public text/map fields; do not expose record labels to tokenizer input.
    data=[dict(instruction=d['instruction'],map_name=d['map_name']) for d in map(json.loads,a.train_jsonl.read_text().splitlines())]
    if len(data)!=21878:raise ValueError('Expected full Revised TRAIN')
    models=[('bert',cfg['text_backbone'],int(cfg['max_text_length'])),('siglip','google/siglip2-large-patch16-384',int(AutoConfig.from_pretrained('google/siglip2-large-patch16-384',local_files_only=True).text_config.max_position_embeddings))]
    result=dict(N=len(data),source_sha256=file_sha256(a.train_jsonl),uses_target_labels=False,encoders={})
    for kind,name,limit in models:
        tokenizer=AutoTokenizer.from_pretrained(name,local_files_only=True);counts=[]
        for at in range(0,len(data),256):
            batch=tokenizer([d['instruction'] for d in data[at:at+256]],truncation=False,padding=False,return_length=True)
            counts.extend(len(x) for x in batch['input_ids'])
        values=np.asarray(counts);longest=np.argsort(values)[-10:][::-1]
        result['encoders'][kind]=dict(name=name,configured_token_limit=limit,truncated_queries=int((values>limit).sum()),truncated_fraction=float((values>limit).mean()),token_length_percentiles={str(q):float(np.percentile(values,q)) for q in [0,25,50,75,90,95,99,100]},
            longest_examples=[dict(index=int(i),tokens=int(values[i]),map_name=data[i]['map_name'],instruction=data[i]['instruction']) for i in longest])
    atomic_json(result,a.output);print(json.dumps({k:{s:v for s,v in d.items() if s!='longest_examples'} for k,d in result['encoders'].items()},indent=2))
if __name__=='__main__':main()
