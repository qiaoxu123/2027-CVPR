"""Evaluate predeclared frozen A3 checkpoints; no fitting or Unseen selection."""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import time
import torch
from graphnav.full_experiment import make_loader,new_model,evaluate
from graphnav.token_language import FrozenBERTTokens
from graphnav.checkpointing import file_sha256,provenance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','seen','unseen','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--batch-size',type=int,default=64);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise ValueError('refusing static evaluation overwrite')
    out.mkdir(parents=True);state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    cfg=SimpleNamespace(**dict(state['args'],batch_size=a.batch_size,workers=a.workers,device='cuda'))
    device=torch.device('cuda');model=new_model(cfg,device).eval().requires_grad_(False);model.load_state_dict(state['model'],strict=True)
    encoder=FrozenBERTTokens(cfg.text_backbone,device='cuda',max_length=getattr(cfg,'max_text_length',96))
    report=dict(checkpoint=str(Path(a.checkpoint).resolve()),checkpoint_sha256=file_sha256(a.checkpoint),epoch=state['epoch'],input_sha256={s:file_sha256(getattr(a,s)) for s in ('seen','unseen')},provenance=provenance(),metrics={},note='Static localization R@k/20m and NE; not closed-loop SR. Deduplicate by source map + instruction with label-independent first occurrence. Frozen checkpoint only; no fitting, epoch selection, or tuning on Unseen.')
    for split in ('seen','unseen'):
        loader=make_loader(getattr(a,split),cfg)
        start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        metrics=evaluate(model,loader,encoder,None,None,cfg,'geometry',device,prediction_path=out/(split+'_predictions.jsonl'))
        torch.cuda.synchronize();metrics.update(seconds=time.perf_counter()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        report['metrics'][split]=metrics;print(json.dumps(dict(split=split,**metrics)),flush=True)
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    rows=[dict(split=s,**m) for s,m in report['metrics'].items()]
    with (out/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


if __name__=='__main__':main()
