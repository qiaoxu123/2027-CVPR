"""SigLIP patch pooling and existing P2 adapter: instance retrieval only.

The identity-trained head has no trained native text-space projection. Thus its
256D vectors are never compared to 1024D SigLIP text and are not semantic scores.
"""
import argparse,json,time
from pathlib import Path
from collections import defaultdict
import torch
from torch.nn import functional as F
from graphnav.landmark_visual_pretrain import FrozenSigLIP2Patches,VisualLandmarkAligner
from experiments.visual_grounding.freeze_dataset import read,sha
from experiments.visual_grounding.statistics import intervals


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--adapter',required=True);p.add_argument('--split',choices=['development','unseen'],default='development');a=p.parse_args();data=Path(a.data);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    result=out/f'patch_retrieval_{a.split}.json'
    if result.exists():raise ValueError('completed evaluation exists')
    index=read(data/'index.jsonl');lookup={r['sample_id']:r for r in index};gallery=read(data/'retrieval.jsonl');ids=sorted({r[k] for r in gallery for k in ['query_sample_id','gallery_sample_id']})
    cache=out/'patch_vectors.pt'
    if cache.exists():
        state=torch.load(cache,map_location='cpu',weights_only=True)
        if state['frozen_sha256']!=sha(data/'frozen.json') or state['adapter_sha256']!=sha(a.adapter):raise ValueError('cache fingerprint changed')
        pooled,adapted=state['pooled'],state['adapter']
    else:
        s=torch.load(a.adapter,map_location='cpu',weights_only=True);args=s['args'];head=VisualLandmarkAligner(vision_dim=1024,hidden=args['hidden'],heads=args['heads']).cuda().eval().requires_grad_(False);head.load_state_dict(s['model'],strict=True)
        backbone=FrozenSigLIP2Patches(device='cuda',microbatch=16,cache_limit=32,cache_directory=None);pp=[];aa=[];t=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        for offset in range(0,len(ids),16):
            paths=[lookup[i]['roi_path'] for i in ids[offset:offset+16]];tokens,mask=backbone.encode(paths)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                mean=(tokens*mask[...,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
                pp.append(F.normalize(mean.float(),dim=-1).cpu());aa.append(F.normalize(head(tokens,mask).float(),dim=-1).cpu())
            if offset%1600==0:print(json.dumps(dict(encoded=offset+len(paths),total=len(ids),seconds=time.perf_counter()-t)),flush=True)
        pooled,adapted=torch.cat(pp),torch.cat(aa)
        state=dict(frozen_sha256=sha(data/'frozen.json'),adapter_sha256=sha(a.adapter),ids=ids,pooled=pooled,adapter=adapted,seconds=time.perf_counter()-t,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,adapter_epoch=s.get('epoch'),note='both representations use actual nonblack observed patches; head is identity-trained; no textual semantic equivalence')
        torch.save(state,cache)
    rows=[r for r in gallery if (r['split']=='val_unseen')==(a.split=='unseen')];groups=defaultdict(list)
    for r in rows:groups[r['split'],r['map_name'],r['object_type']].append(r)
    il={s:i for i,s in enumerate(ids)};pred=[]
    for name,vectors in [('mean_patch',pooled),('old_p2_adapter',adapted)]:
        for key,rs in sorted(groups.items()):
            if len(rs)<2:continue
            sims=vectors[[il[r['query_sample_id']] for r in rs]]@vectors[[il[r['gallery_sample_id']] for r in rs]].T
            for i,r in enumerate(rs):
                rank=sims[i].argsort(descending=True).tolist()
                for k in [1,5]:pred.append(dict(model=name,split=key[0],map_name=key[1],object_type=key[2],object_id=r['object_id'],sample_id=r['query_sample_id'],k=k,correct=i in rank[:k],candidate_size=len(rs)))
    metrics={}
    for name in ['mean_patch','old_p2_adapter']:
        metrics[name]={}
        for split in sorted(set(r['split'] for r in pred)):
            metrics[name][split]={}
            for k in [1,5]:
                rs=[r for r in pred if r['model']==name and r['split']==split and r['k']==k];metrics[name][split][f'R@{k}']=intervals([r['correct'] for r in rs],[r['map_name'] for r in rs])
    (out/f'patch_retrieval_predictions_{a.split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in pred));result.write_text(json.dumps(dict(metrics=metrics,seconds=state['seconds'],peak_allocated_gib=state['peak_allocated_gib'],adapter_sha256=sha(a.adapter),limitation=state['note']),indent=2));print(json.dumps(metrics))
if __name__=='__main__':main()
