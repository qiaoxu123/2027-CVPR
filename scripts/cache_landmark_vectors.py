"""Freeze the selected P2 head on audited actual Train/Seen ROI pixels.

No goal labels or future-prefix assembly is performed here. Source episode and
observation steps remain metadata for subsequent causal history construction.
"""
import argparse
import json
from pathlib import Path
import time
import torch
from graphnav.landmark_visual_pretrain import FrozenSigLIP2Patches,VisualLandmarkAligner
from graphnav.checkpointing import file_sha256,atomic_save,provenance


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('rois','checkpoint','output','patch-cache'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--batch-size',type=int,default=32);args=parser.parse_args();out=Path(args.output)
    if out.exists():raise ValueError('refusing frozen landmark-vector cache overwrite')
    out.mkdir(parents=True);state=torch.load(args.checkpoint,map_location='cpu',weights_only=True);config=state['args']
    backbone=FrozenSigLIP2Patches(config['image_backbone'],device='cuda',cache_limit=256,
        microbatch=config['image_microbatch'],cache_directory=args.patch_cache)
    head=VisualLandmarkAligner(vision_dim=backbone.hidden_size,hidden=config['hidden'],heads=config['heads']).cuda().eval().requires_grad_(False)
    head.load_state_dict(state['model'],strict=True)
    rows=[json.loads(line) for line in Path(args.rois).read_text().splitlines() if line]
    for row in rows:
        if row['observed_step']>row['observation_cutoff_step']:raise ValueError('future ROI rejected')
    result=[];start=time.perf_counter();source=provenance()
    with torch.no_grad():
        for first in range(0,len(rows),args.batch_size):
            paths=[row['roi_path'] for row in rows[first:first+args.batch_size]]
            tokens,mask=backbone.encode(paths)
            with torch.autocast('cuda',dtype=torch.bfloat16):vectors=head(tokens,mask)
            if not torch.isfinite(vectors).all():raise ValueError('nonfinite cached landmark vector')
            result.append(vectors.half().cpu())
            if (first//args.batch_size)%25==0:print(json.dumps(dict(completed=first+len(paths),total=len(rows),seconds=time.perf_counter()-start)),flush=True)
    atomic_save(torch.cat(result),out/'vectors.pt')
    (out/'rows.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    report=dict(complete=True,rows=len(rows),seconds=time.perf_counter()-start,
        newly_encoded_images=backbone.encoded_images,disk_hits=backbone.disk_hits,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,checkpoint_sha256=file_sha256(args.checkpoint),
        source_rois_sha256=file_sha256(args.rois),vectors_sha256=file_sha256(out/'vectors.pt'),
        provenance=source,note='Frozen current-pixel vectors; no destination labels or online future gallery.')
    (out/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
