"""Original4m observed-support Gaussians using stable direct world differences."""
import math
import torch
from torch.nn import functional as F


def candidate_kernels(spatial,*,sigma=4.):
    if not math.isfinite(sigma) or sigma<=0:raise ValueError('Invalid observed kernel sigma')
    xy=spatial['candidate_xy'].float();grid=spatial['grid_world'].float();support=spatial['candidate_support'].bool()
    if xy.ndim!=3 or grid.ndim!=3 or xy.shape[0]!=grid.shape[0] or xy.shape[-1]!=2 or grid.shape[-1]!=2 or support.shape!=(*xy.shape[:2],grid.shape[1]):raise ValueError('Observed kernel axes changed')
    if not torch.isfinite(xy).all() or not torch.isfinite(grid).all():raise ValueError('Nonfinite public world geometry')
    # Avoid ||x||²+||y||²-2x.y cancellation at large map coordinates. The
    # mathematical sigma/support are unchanged; no private target is involved.
    d2=(xy[:,:,None]-grid[:,None]).square().sum(-1)
    logits=(-d2/(2*sigma*sigma)).masked_fill(~support,-torch.inf)
    safe=torch.where(support.any(-1,keepdim=True),logits,torch.zeros_like(logits))
    return safe.softmax(-1)*support


def proposal_with_quality(spatial,scores,trust,reliable):
    kernels=candidate_kernels(spatial);eligible=reliable.bool()&spatial['candidate_support'].bool().any(-1)
    if scores.shape!=eligible.shape or trust.shape!=scores.shape:raise ValueError('Evidence proposal axes changed')
    available=eligible.any(-1)
    logits=(scores.float()+F.logsigmoid(trust.float())).masked_fill(~eligible,-torch.inf)
    safe=torch.where(available[:,None],logits,torch.zeros_like(logits))
    weights=safe.softmax(-1)*eligible
    return (weights[:,:,None]*kernels).sum(1),available,(weights*trust.float().sigmoid()).sum(-1)


def proposal(spatial,scores,trust,reliable):
    q,available,_=proposal_with_quality(spatial,scores,trust,reliable)
    return q,available


def spatial_utility_labels(spatial,target_xy):
    """PRIVATE offline spatial suitability, never an object identity assertion."""
    kernels=candidate_kernels(spatial);grid=spatial['grid_world'].float()
    if target_xy.shape!=(len(grid),2) or not torch.isfinite(target_xy).all():raise ValueError('Invalid private target coordinates')
    distance=(grid-target_xy.float()[:,None]).norm(dim=-1)
    near=distance.gather(1,kernels.argmax(-1))<=20.
    return near&spatial['candidate_support'].bool().any(-1)
