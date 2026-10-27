"""Bounded local probability mixing: preserve outside probability and local mass."""
import torch


def local_mix(base,grid,xy,scores,support,alpha):
    if not 0<=float(alpha)<=.5:raise ValueError('alpha must lie in [0,.5]')
    empty=dict(logits=base,support=torch.zeros_like(base,dtype=torch.bool),changed=False)
    if not len(xy) or float(alpha)==0:return empty
    support=support.bool();usable=support.any(-1)
    if not usable.any():return empty
    support=support[usable];xy=xy[usable];scores=scores[usable]
    union=support.any(0)
    kernels=torch.exp(-.5*(torch.cdist(grid.float(),xy.float()).T/4).square())*support
    # Avoid complete underflow when the coarse center is distant from the box.
    kernels=kernels+support.to(kernels.dtype)*1e-12
    kernels=kernels/kernels.sum(-1,keepdim=True).clamp_min(1e-30)
    q=(scores.float().softmax(0)[:,None]*kernels).sum(0)
    shift=base.float().max();raw=(base.float()-shift).exp();mass=raw[union].sum()
    if mass<=0:return empty
    mixed=raw.clone();mixed[union]=(1-float(alpha))*raw[union]+float(alpha)*mass*q[union]
    logits=base.float().clone();logits[union]=shift+mixed[union].clamp_min(1e-38).log()
    return dict(logits=logits,support=union,changed=not torch.equal(logits,base))
