"""Privileged OFFLINE diagnosis only. No import by normal inference modules."""
from dataclasses import replace
import torch

WEIGHTS=(0.5,1.,2.,4.)


def rerank(logits, grid_world, anchors_world, angles, mode, weight=1., cell_diagonal=20.):
    """No target coordinates/ranges accepted; angles already frozen offline.

    Every already-bound anchor has equal weight; none selected by target GT.
    Inputs CPU or GPU FP32. Nonexistent support falls back unchanged.
    """
    z=logits.float()
    if mode=='A' or len(angles)==0:return z.clone()
    if mode not in ('B','C','D'):raise ValueError('Unknown Oracle diagnostic')
    a=torch.as_tensor(anchors_world,device=z.device,dtype=torch.float32).reshape(-1,2)
    t=torch.as_tensor(angles,device=z.device,dtype=torch.float32).reshape(-1)
    xy=torch.as_tensor(grid_world,device=z.device,dtype=torch.float32)
    if xy.shape!=(z.numel(),2) or len(a)!=len(t) or not torch.isfinite(t).all() or cell_diagonal<=0:raise ValueError('Bad diagnostic geometry')
    d=xy[:,None,:]-a[None,:,:];r=d.norm(dim=-1)
    unit=torch.stack((t.cos(),t.sin()),-1)
    along=(d*unit).sum(-1)
    cosine=(along/r.clamp_min(1e-6)).mean(-1)
    # An anchor-center candidate has undefined angle, never perfect alignment.
    cosine=cosine.masked_fill((r<1e-6).any(-1),-1.)
    if mode=='D':return z+float(weight)*4.*cosine.reshape_as(z)
    if mode=='B':score=cosine
    else:
        supported=(along>=0).all(-1)&(r>=1e-6).all(-1)
        if not supported.any():return z.clone()
        perp=(d[...,0]*unit[None,:,1]-d[...,1]*unit[None,:,0]).abs().mean(-1)
        score=(-perp/float(cell_diagonal)).masked_fill(~supported,-1e4)
    # Lexicographic score then original Belief for exact ties. Assign monotonic
    # scores; original-logit amplitude cannot alter geometric ordering.
    base=torch.argsort(z.flatten(),descending=True,stable=True)
    order=base[torch.argsort(score[base],descending=True,stable=True)]
    # Preserve physical scoring scale for the controller, with a fixed <=1e-4 prior-rank tie tolerance.
    tie=torch.empty_like(score)
    tie[base]=torch.arange(len(base),0,-1,device=z.device,dtype=torch.float32)/len(base)
    return (score+1e-4*tie).reshape_as(z)


def replace_field(field, scores, grid):
    if field.dense_context is None:raise ValueError('Dense context required for honest candidate realignment')
    ids=scores.topk(field.candidate_ids.shape[-1],dim=-1).indices
    return replace(field,belief_logits=scores,candidate_ids=ids,
        candidate_xy=grid.gather(1,ids[...,None].expand(-1,-1,2)),
        candidate_context=field.dense_context.gather(1,ids[...,None].expand(-1,-1,field.dense_context.shape[-1])))
