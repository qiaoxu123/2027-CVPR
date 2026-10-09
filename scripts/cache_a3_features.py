"""Cache frozen A3 outputs from public queries, without GT or repeated BERT encoding."""
import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
import time
import torch
from torch.utils.data import DataLoader,Subset
from graphnav.candidate_geometry import FullCandidateDataset,collate_full
from graphnav.full_experiment import _batch_inputs,new_model
from graphnav.token_language import FrozenBERTTokens
from graphnav.checkpointing import atomic_save,file_sha256,provenance


def dense_cache_values(field):
    # Dense goal scores preserve FP32 ranking; FP16 can create artificial ties.
    return dict(dense_context=field.dense_context.half().cpu(),belief_logits=field.belief_logits.float().cpu())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queries',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True);p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--chunk-queries',type=int,default=512)
    p.add_argument('--dense-context',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--bert-cache',help='Reuse fingerprinted frozen tokens; no repeated text encoding across seeds')
    p.add_argument('--device',default='cuda');args=p.parse_args();out=Path(args.output)
    if out.exists() and not args.resume:raise ValueError('refusing cache overwrite; use --resume')
    out.mkdir(parents=True,exist_ok=True)
    protocol=dict(query_sha256=file_sha256(args.queries),checkpoint_sha256=file_sha256(args.checkpoint),
        dense_context=args.dense_context,dense_logits_dtype='float32' if args.dense_context else None,
        precision='bf16' if args.device=='cuda' else 'fp32')
    if args.bert_cache:protocol['bert_cache_sha256']=file_sha256(args.bert_cache)
    status=out/'manifest.json'
    manifest=json.loads(status.read_text()) if status.exists() else dict(protocol=protocol,chunks=[],completed_queries=0,provenance=provenance())
    if manifest['protocol']!=protocol:raise ValueError('cache source/model/protocol fingerprint mismatch')
    for chunk in manifest['chunks']:
        if file_sha256(out/chunk['file'])!=chunk['sha256']:raise ValueError('cache chunk SHA256 mismatch')
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True);config=SimpleNamespace(**state['args'])
    device=torch.device(args.device);model=new_model(config,device).eval().requires_grad_(False)
    model.load_state_dict(state['model'],strict=True)
    encoder=FrozenBERTTokens(config.text_backbone,device=args.device,max_length=getattr(config,'max_text_length',96))
    if args.bert_cache:encoder.load_cache(args.bert_cache)
    dataset=FullCandidateDataset(args.queries,grid_size=config.grid_size,max_landmarks=config.max_landmarks,
        max_roads=config.max_roads,max_relations=config.max_relations,prior_sigma_m=config.prior_sigma_m,
        rule_sigma_m=config.rule_sigma_m,deduplicate_queries=False)
    # Public query files may contain multiple reference priors for the same text.
    # Keep their exact source-index order; no destination/ID-based query selection.
    if any(row['target_xy']!=[0.,0.] for row in dataset.records):
        raise ValueError('feature cache requires public queries with dummy labels only')
    start=manifest['completed_queries'];pending={};pending_count=0;timer=time.perf_counter()
    loader=DataLoader(Subset(dataset,range(start,len(dataset))),batch_size=args.batch_size,
        num_workers=args.workers,persistent_workers=args.workers>0,collate_fn=collate_full,
        pin_memory=device.type=='cuda')
    def save_chunk():
        nonlocal pending,pending_count
        if not pending_count:return
        file=f"chunk_{len(manifest['chunks']):04d}.pt"
        atomic_save({key:torch.cat(parts) for key,parts in pending.items()},out/file)
        manifest['chunks'].append(dict(file=file,first_query=manifest['completed_queries'],queries=pending_count,
            sha256=file_sha256(out/file)))
        manifest['completed_queries']+=pending_count
        temporary=status.with_suffix('.tmp');temporary.write_text(json.dumps(manifest,indent=2));temporary.replace(status)
        print(json.dumps(dict(completed=manifest['completed_queries'],total=len(dataset),seconds=time.perf_counter()-timer)),flush=True)
        pending={};pending_count=0
    with torch.no_grad():
        for batch in loader:
            observable,tokens,mask,names=_batch_inputs(batch,encoder,device,'geometry')
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.type=='cuda'):
                field=model(observable,variant='geometry',tokens=tokens,token_mask=mask,node_emb=names)
            probabilities=field.belief_logits.float().softmax(-1)
            values=dict(candidate_xy=field.candidate_xy.float().cpu(),
                candidate_context=field.candidate_context.half().cpu(),
                candidate_logits=field.belief_logits.gather(1,field.candidate_ids).float().cpu(),
                belief_entropy=(-(probabilities*probabilities.clamp_min(1e-8).log()).sum(-1)/math.log(probabilities.shape[-1])).cpu(),
                bounds=torch.tensor(batch['world_bounds'],dtype=torch.float32))
            if args.dense_context:
                values.update(dense_cache_values(field))
            for key,value in values.items():pending.setdefault(key,[]).append(value)
            pending_count+=len(batch['text'])
            if pending_count>=args.chunk_queries:save_chunk()
    save_chunk();manifest.update(complete=True,seconds=time.perf_counter()-timer,total_queries=len(dataset))
    status.write_text(json.dumps(manifest,indent=2));encoder.save_cache(out/'bert_tokens.pt')


if __name__=='__main__':main()
