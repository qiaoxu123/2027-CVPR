"""Frozen TRAIN-only posthoc correction-amplitude analysis; no model selection."""
import argparse,json
from pathlib import Path
import torch
from scripts.unified_pgvb_v3_pilot import V3CacheDataset,fit_dev_partition,_collate,_load_base
from graphnav.unified_pgvb_v3 import UnifiedPGVBV3
from graphnav.v3_diagnostics import visual_correction_capacity
from graphnav.checkpointing import atomic_json,file_sha256

@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    for k in ['source-v2','checkpoint','cache','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();ds=V3CacheDataset(a.cache);_,dev=fit_dev_partition(ds,42)
    geom,_=_load_base(a.source_v2,'cuda');model=UnifiedPGVBV3(geom).cuda().eval();ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True);model.load_state_dict(ck['model'],strict=True)
    rows=[]
    for i in dev:
        public,labels=ds[i];kw,private=_collate([(public,labels)],'cuda');out=model(**kw)
        r=visual_correction_capacity(out,private['target_xy'],kw['world_bounds'],model.visual_gain.tanh())[0]
        rows.append(dict(index=i,episode_id=labels['episode_id'],**r))
    wrong=[x for x in rows if x['region_defined'] and not x['geometry_correct']]
    summary=dict(N=len(rows),geometry_wrong=len(wrong),amplitude=float(model.visual_gain.tanh().abs()),
        supported_wrong=sum(x['correct_region_observed'] for x in wrong),
        supported_but_bound_insufficient=sum(x['correct_region_observed'] and x['bound_insufficient'] for x in wrong),
        visual_region_top1_correct_but_bound_insufficient=sum(bool(x['visual_only_top1_correct_region_proxy']) and x['bound_insufficient'] for x in wrong),
        capacity_tie_undetermined=sum(x['capacity_tie_undetermined'] for x in wrong),
        capacity_sufficient_but_fusion_failed=sum(x['can_flip_under_independent_bounded_updates'] and not x['fused_correct'] for x in wrong),
        corrected=sum(x['fused_correct'] for x in wrong),checkpoint_sha256=file_sha256(a.checkpoint),
        scope='TRAIN pilot heldout, already analysed; not independent confirmation; spatial proxy not instance accuracy')
    atomic_json(dict(summary=summary,rows=rows),a.output);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
