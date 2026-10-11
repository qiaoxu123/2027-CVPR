"""Initial-static localization only; no observation-time belief or navigation claims."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader
from graphnav.candidate_geometry import FullCandidateDataset,collate_full
from graphnav.full_experiment import _batch_inputs,new_model
from graphnav.checkpointing import file_sha256,provenance
from graphnav.stage1_runtime import retrieval_ids
from graphnav.token_language import FrozenBERTTokens


def main():
    p=argparse.ArgumentParser()
    for key in ('checkpoint','records','source-a3','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();output=Path(a.output)
    if output.exists():raise ValueError('refusing static metrics overwrite')
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    complete='a3_args'in state;cfg=SimpleNamespace(**(state['a3_args']if complete else state['args']))
    model=new_model(cfg,torch.device('cuda')).eval()
    weights={k[len('a3.'):]:v for k,v in state['model'].items()if k.startswith('a3.')}if complete else state['model']
    model.load_state_dict(weights,strict=True)
    source=torch.load(a.source_a3,map_location='cpu',weights_only=True)
    encoder=FrozenBERTTokens(cfg.text_backbone,device='cuda',max_length=getattr(cfg,'max_text_length',96))
    train_cache=source['input_fingerprint']['train_bert_cache'];seen_cache=source['input_fingerprint']['seen_bert_cache']
    for cache in (train_cache,seen_cache):
        if file_sha256(cache['path'])!=cache['sha256']:raise ValueError('static BERT cache SHA mismatch')
    encoder.load_cache(train_cache['path']);train=encoder.cache.copy();encoder.load_cache(seen_cache['path']);encoder.cache.update(train);del train
    data=FullCandidateDataset(a.records,grid_size=cfg.grid_size,max_landmarks=cfg.max_landmarks,max_roads=cfg.max_roads,max_relations=cfg.max_relations,prior_sigma_m=cfg.prior_sigma_m,rule_sigma_m=cfg.rule_sigma_m)
    loader=DataLoader(data,batch_size=16,num_workers=4,collate_fn=collate_full)
    output.parent.mkdir(parents=True,exist_ok=True);count=0;totals={};index=0
    with torch.inference_mode(),output.with_suffix('.predictions.jsonl').open('w')as stream:
        for batch in loader:
            x,t,m,n=_batch_inputs(batch,encoder,torch.device('cuda'),'geometry')
            with torch.autocast('cuda',dtype=torch.bfloat16):field=model(x,variant='geometry',tokens=t,token_mask=m,node_emb=n)
            ids=retrieval_ids(field.belief_logits);xy=x['grid_xy'].gather(1,ids[...,None].expand(-1,-1,2))
            dist=((xy-batch['target_xy'].to('cuda')[:,None])*batch['map_sizes'].to('cuda')[:,None]).norm(dim=-1).cpu()
            for distances in dist.tolist():
                row=dict(index=index,map_name=data.records[index]['map_name'],distance_m=distances,NE_m=distances[0],**{f'R@{k}/20m':float(min(distances[:k])<=20)for k in (1,4,8,16)})
                stream.write(json.dumps(row)+'\n');index+=1
                for key,value in row.items():
                    if key=='NE_m'or key.startswith('R@'):totals[key]=totals.get(key,0.)+value
            count+=len(dist)
    report=dict(samples=count,metrics={k:v/count for k,v in totals.items()},checkpoint_sha256=file_sha256(a.checkpoint),input_sha256=file_sha256(a.records),provenance=provenance(),
        definition='Initial static goal field, deduplicated map+instruction. FP32 legacy topk(8) preserved and extended to16. Ground-truth read after forward; not dynamic observed belief or closed-loop SR.')
    output.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
