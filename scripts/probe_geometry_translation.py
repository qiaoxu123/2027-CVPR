"""TRAIN Geometry translation counterfactual; never chooses a helpful offset."""
import argparse
import json
import re
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from graphnav.candidate_geometry import collate_full
from graphnav.checkpointing import atomic_json, file_sha256
from graphnav.full_experiment import _batch_inputs
from graphnav.geometric_translation_probe import translate_observable, average_aligned_logits
from graphnav.local_goal_decoder import decode_first_candidate
from graphnav.token_language import FrozenBERTTokens
from scripts.train_v2_top1_geometry import Indexed
from scripts.unified_pgvb_v3_pilot import _load_base


def decode(field, grid, valid):
    logits=field.belief_logits.float().masked_fill(~valid,-torch.inf)
    ids=logits.topk(field.candidate_ids.shape[-1],dim=-1).indices
    masked=replace(field,belief_logits=logits,candidate_ids=ids,
                   candidate_xy=grid.gather(1,ids[...,None].expand(-1,-1,2)),
                   candidate_context=field.dense_context.gather(1,ids[...,None].expand(-1,-1,field.dense_context.shape[-1])))
    soft=decode_first_candidate(masked,grid)
    return dict(grid=masked.candidate_xy[:,0],local=soft.candidate_xy[:,0]),ids[:,0]


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','source','trained','bert-cache','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--limit',type=int,default=0)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model,cfg=_load_base(a.source,'cuda')
    weights=torch.load(a.trained,map_location='cpu',weights_only=True)['model']
    model.load_state_dict({k[9:]:v for k,v in weights.items() if k.startswith('geometry.')},strict=True)
    model.eval().requires_grad_(False)
    encoder=FrozenBERTTokens(cfg.text_backbone,device='cuda',max_length=96);encoder.load_cache(a.bert_cache)
    ds=Indexed(a.data/'public_train.jsonl',grid_size=28,max_landmarks=cfg.max_landmarks,
               max_roads=cfg.max_roads,max_relations=cfg.max_relations,
               prior_sigma_m=cfg.prior_sigma_m,rule_sigma_m=cfg.rule_sigma_m,deduplicate_queries=False)
    assert all(r['target_xy']==[0.,0.] for r in ds.records)
    labels=json.loads((a.data/'labels_private.json').read_text())
    ids=[r['index'] for r in labels if r['split']=='development']
    if a.limit:ids=ids[:a.limit]
    shifts={'x_plus':(.1,0),'x_minus':(-.1,0),'y_plus':(0,.1),'y_minus':(0,-.1)}
    manifest=dict(checkpoint_sha256=file_sha256(a.trained),source_sha256=file_sha256(a.source),
                  data_sha256=file_sha256(a.data/'manifest.json'),bert_cache_sha256=file_sha256(a.bert_cache),
                  shifts=shifts,relative_position=bool(model.relative_position),n=len(ids),
                  aggregation='Original plus four equally weighted translated fields; per-branch common-query additive gauge alignment; unknown queries omitted, never negative',
                  script_sha256=file_sha256(__file__),helper_sha256=file_sha256('graphnav/geometric_translation_probe.py'),
                  precision='FP32; TF32 off',scope='TRAIN development only; geometric physical-scene translations, no RGB or navigation; no GT offset selection')
    atomic_json(manifest,a.output/'manifest.json')
    rows=[];average_rows=[];start=time.perf_counter()
    for batch in DataLoader(ds,batch_size=16,sampler=ids,num_workers=0,collate_fn=collate_full):
        obs,tokens,mask,names=_batch_inputs(batch,encoder,'cuda','geometry')
        base=model(obs,variant='geometry',tokens=tokens,token_mask=mask,node_emb=names)
        ii=batch['row_index'].tolist()
        bounds=torch.tensor([ds.records[i]['world_bounds'] for i in ii],device='cuda')
        branches=[]
        for key,offset in shifts.items():
            translated,common=translate_observable(obs,offset)
            shifted=model(translated,variant='geometry',tokens=tokens,token_mask=mask,node_emb=names)
            branches.append((shifted.belief_logits,common))
            original_xy,original_id=decode(base,obs['grid_xy'],common)
            shifted_xy,shifted_id=decode(shifted,translated['grid_xy'],common)
            delta=torch.tensor(offset,device='cuda')
            diff=shifted.belief_logits-base.belief_logits
            mean=(diff*common).sum(-1)/common.sum(-1)
            centered=((diff-mean[:,None]).abs()*common).sum(-1)/common.sum(-1)
            # All public predictions complete; labels only evaluate translation back.
            target=torch.tensor([labels[i]['target_world'] for i in ii],device='cuda')
            for b,i in enumerate(ii):
                text=ds.records[i]['instruction']
                clean=bool(obs['entity_valid'][b].any()) and not bool(re.search(r'\b(map|image|picture|screen|frame|boundary|corner|edge|upper|lower|top|bottom)\b',text,re.I))
                row=dict(index=i,shift=key,anchor_clean=clean,top1_index_changed=bool(original_id[b]!=shifted_id[b]),
                         centered_logit_change=float(centered[b]),n_common=int(common[b].sum()))
                for decoder in ('grid','local'):
                    before=bounds[b,:2]+original_xy[decoder][b]*(bounds[b,2:]-bounds[b,:2])
                    after=bounds[b,:2]+(shifted_xy[decoder][b]-delta)*(bounds[b,2:]-bounds[b,:2])
                    row[decoder]=dict(base_NE=float((before-target[b]).norm()),shifted_NE=float((after-target[b]).norm()),
                                      equivariance_error_m=float((after-before).norm()))
                rows.append(row)
        averaged=replace(base,belief_logits=average_aligned_logits(base.belief_logits,branches))
        available=torch.ones_like(base.belief_logits,dtype=torch.bool)
        base_xy,_=decode(base,obs['grid_xy'],available)
        mean_xy,_=decode(averaged,obs['grid_xy'],available)
        for b,i in enumerate(ii):
            row=dict(index=i)
            for decoder in ('grid','local'):
                before=bounds[b,:2]+base_xy[decoder][b]*(bounds[b,2:]-bounds[b,:2])
                after=bounds[b,:2]+mean_xy[decoder][b]*(bounds[b,2:]-bounds[b,:2])
                row[decoder]=dict(base_NE=float((before-target[b]).norm()),mean_NE=float((after-target[b]).norm()))
            average_rows.append(row)
        if len(rows)%1024==0:print(json.dumps(dict(rows=len(rows),seconds=time.perf_counter()-start)),flush=True)
    result={}
    for subset in ('all','anchor_clean'):
        result[subset]={}
        for key in shifts:
            rr=[r for r in rows if r['shift']==key and (subset=='all' or r['anchor_clean'])]
            metrics={}
            for decoder in ('grid','local'):
                before=np.array([r[decoder]['base_NE'] for r in rr]);after=np.array([r[decoder]['shifted_NE'] for r in rr]);drift=np.array([r[decoder]['equivariance_error_m'] for r in rr])
                metrics[decoder]=dict(n=len(rr),base_R20=float((before<=20).mean()),shifted_R20=float((after<=20).mean()),
                                      base_NE=float(before.mean()),shifted_NE=float(after.mean()),
                                      corrections=int(((before>20)&(after<=20)).sum()),damage=int(((before<=20)&(after>20)).sum()),
                                      drift_median=float(np.median(drift)),drift_P95=float(np.percentile(drift,95)))
            result[subset][key]=dict(metrics=metrics,top1_changes=sum(r['top1_index_changed'] for r in rr),
                                     mean_centered_logit_change=float(np.mean([r['centered_logit_change'] for r in rr])))
    average_summary={}
    for decoder in ('grid','local'):
        before=np.array([r[decoder]['base_NE'] for r in average_rows]);after=np.array([r[decoder]['mean_NE'] for r in average_rows])
        average_summary[decoder]=dict(n=len(before),base_R20=float((before<=20).mean()),mean_R20=float((after<=20).mean()),
                                     base_NE=float(before.mean()),mean_NE=float(after.mean()),
                                     corrections=int(((before>20)&(after<=20)).sum()),damage=int(((before<=20)&(after>20)).sum()))
    atomic_json(average_rows,a.output/'average_rows.json')
    atomic_json(rows,a.output/'rows.json');atomic_json(dict(summary=result,fixed_average=average_summary,seconds=time.perf_counter()-start),a.output/'summary.json')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
