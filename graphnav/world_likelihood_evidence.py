"""Observation likelihood on one world grid; no second copy of the prior mass."""
import math
import torch


def world_evidence_logits(prior_logits, local_uv, scores, local_valid):
    cells=prior_logits.shape[-1];grid=math.isqrt(cells)
    if grid*grid!=cells or local_uv.shape!=(*scores.shape,2) or scores.shape!=local_valid.shape:
        raise ValueError('square world grid and matching evidence required')
    if local_valid.dtype!=torch.bool or not torch.isfinite(local_uv).all():
        raise ValueError('finite public coordinates and bool validity required')
    valid=local_valid & (local_uv>=0).all(-1) & (local_uv<=1).all(-1)
    scores=scores.float().masked_fill(~valid,0)
    xy=(local_uv.float()*grid-.5).clamp(0,grid-1)
    x0=xy[...,0].floor().long();y0=xy[...,1].floor().long()
    x1=(x0+1).clamp(max=grid-1);y1=(y0+1).clamp(max=grid-1)
    wx=xy[...,0]-x0;wy=xy[...,1]-y0
    numerator=torch.zeros_like(prior_logits,dtype=torch.float32);denominator=torch.zeros_like(numerator)
    for x,y,weight in ((x0,y0,(1-wx)*(1-wy)),(x1,y0,wx*(1-wy)),(x0,y1,(1-wx)*wy),(x1,y1,wx*wy)):
        weight=weight*valid
        numerator=numerator.scatter_add(1,y*grid+x,scores*weight)
        denominator=denominator.scatter_add(1,y*grid+x,weight)
    # Average log-likelihood, not a sum of duplicated prior probabilities.
    evidence=numerator/denominator.clamp_min(1e-8)
    return prior_logits.detach().float().log_softmax(-1)+evidence
