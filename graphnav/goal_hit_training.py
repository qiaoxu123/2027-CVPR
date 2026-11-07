"""TRAIN-only world-distance supervision; no target-dependent inference operations."""
import torch
import torch.nn.functional as F
from .static_localization import cross_entropy_goal


def goal_hit_loss(logits, grid_xy, bounds, target_world, target_xy, mode):
    z=logits.float();world=bounds[:,None,:2]+grid_xy.float()*(bounds[:,None,2:]-bounds[:,None,:2]);distance=(world-target_world[:,None]).norm(dim=-1)
    if mode=='legacy':return cross_entropy_goal(z,target_xy,grid_size=28),dict(no_positive20=int((distance.min(-1).values>20).sum()))
    soft=(-distance.square()/(2*8.**2)).softmax(-1);ce=-(soft*z.log_softmax(-1)).sum(-1).mean()
    inside=distance<=20;valid=inside.any(-1);outside=(~inside).any(-1);usable=valid&outside
    mass=ce.new_zeros(());margin=ce.new_zeros(())
    if valid.any():mass=-torch.logsumexp(z[valid].log_softmax(-1).masked_fill(~inside[valid],-torch.inf),-1).mean()
    if usable.any():
        pos=z.masked_fill(~inside,-torch.inf).max(-1).values;neg=z.masked_fill(inside,-torch.inf).max(-1).values;margin=F.relu(.5+neg[usable]-pos[usable]).mean()
    if mode=='ce8':loss=ce
    elif mode=='hit20':loss=ce+.5*mass+.1*margin
    else:raise ValueError('Unknown loss mode')
    return loss,dict(ce8=float(ce.detach()),mass20=float(mass.detach()),hard_margin=float(margin.detach()),no_positive20=int((~valid).sum()))
