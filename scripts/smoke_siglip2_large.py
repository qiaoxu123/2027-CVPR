"""Real Large/384 patch API, finite loss, head gradient and frozen backbone smoke."""
import argparse
import json
from pathlib import Path
import time

import torch

from graphnav.landmark_visual_pretrain import (
    FrozenSigLIP2Patches, PairedROIs, VisualLandmarkAligner, roi_collate, match_loss)
from graphnav.landmark_visual_train import encode_twoview


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rois',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--cache-directory',required=True)
    args=p.parse_args()
    torch.manual_seed(0)
    ds=PairedROIs(args.rois,deterministic=True)
    selected=[i for i,(_,rows) in enumerate(ds.groups) if rows[0]['object_type']=='Building'][:4]
    batch=roi_collate([ds[i] for i in selected])
    t=time.perf_counter()
    backbone=FrozenSigLIP2Patches(device='cuda',microbatch=8,
                                 cache_directory=args.cache_directory)
    net=VisualLandmarkAligner(vision_dim=backbone.hidden_size,hidden=256,heads=8).cuda()
    patches,valid=backbone.encode(batch['first'])
    assert patches.shape==(len(selected),576,1024)
    assert torch.isfinite(patches).all() and valid.any(1).all()
    a,b=encode_twoview(batch,backbone,net,torch.device('cuda'))
    loss,parts=match_loss(net,a,b,batch['type'].cuda(),torch.randn(len(ds.types),768,device='cuda'))
    loss.backward()
    gradient=float(net.patch[0].weight.grad.abs().sum())
    assert torch.isfinite(loss) and gradient>0
    assert all(not p.requires_grad and p.grad is None for p in backbone.model.parameters())
    report={'backbone':'google/siglip2-large-patch16-384','revision':backbone.protocol[1],
            'patch_shape':list(patches.shape),'vision_hidden':backbone.hidden_size,
            'valid_patch_fraction':float(valid.float().mean()),
            'loss':float(loss.detach()),'loss_components':parts,'head_gradient_L1':gradient,
            'all_backbone_parameters_frozen':True,'bf16':True,
            'seconds':time.perf_counter()-t,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,
            'actual_observed_ROIs':batch['first']+batch['second'],'passed':True}
    Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
