"""Bounded real-label A3 backward throughput; weights are never saved or promoted."""
import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from graphnav.candidate_geometry import FullCandidateDataset,collate_full
from graphnav.full_experiment import new_model,_batch_inputs
from graphnav.token_language import FrozenBERTTokens
from graphnav.static_localization import cross_entropy_goal
from graphnav.checkpointing import provenance,file_sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','data','bert-cache','output'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--batch-size',type=int,required=True);parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();out=Path(args.output)
    if out.exists():raise ValueError('refusing benchmark overwrite')
    out.mkdir(parents=True);source=provenance();torch.manual_seed(0)
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=True);cfg=SimpleNamespace(**payload['args'])
    device=torch.device('cuda');model=new_model(cfg,device);model.load_state_dict(payload['model']);model.train()
    encoder=FrozenBERTTokens(cfg.text_backbone,device='cuda',max_length=getattr(cfg,'max_text_length',96));encoder.load_cache(args.bert_cache)
    dataset=FullCandidateDataset(args.data,grid_size=cfg.grid_size,max_landmarks=cfg.max_landmarks,
        max_roads=cfg.max_roads,max_relations=cfg.max_relations,prior_sigma_m=cfg.prior_sigma_m,rule_sigma_m=cfg.rule_sigma_m)
    indices=np.random.default_rng(0).choice(len(dataset),1024,replace=False).tolist()
    loader=DataLoader(Subset(dataset,indices),batch_size=args.batch_size,num_workers=args.workers,
        persistent_workers=args.workers>0,pin_memory=True,collate_fn=collate_full)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4);iterator=iter(loader)
    def update(batch):
        inputs,tokens,mask,names=_batch_inputs(batch,encoder,device,'geometry');optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):field=model(inputs,variant='geometry',tokens=tokens,token_mask=mask,node_emb=names)
        loss=cross_entropy_goal(field.belief_logits,batch['target_xy'].to(device),grid_size=cfg.grid_size)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
    for _ in range(2):update(next(iterator))
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();iterator=iter(loader);started=time.perf_counter()
    samples=0
    for batch in iterator:update(batch);samples+=len(batch['text'])
    torch.cuda.synchronize();seconds=time.perf_counter()-started
    result=dict(args=vars(args),provenance=source,checkpoint_sha256=file_sha256(args.checkpoint),data_sha256=file_sha256(args.data),
        samples=samples,seconds=seconds,samples_per_second=samples/seconds,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        protocol='BF16 real Revised Train Seen labels; frozen cached BERT96; 2 warmup updates then same1024 queries; workers4. Private benchmark weight copy discarded; not accuracy evidence.')
    (out/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
