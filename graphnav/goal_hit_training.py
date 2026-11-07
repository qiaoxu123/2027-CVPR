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


def fidelity_hit_loss(logits, teacher_logits, grid_xy, bounds, target_world):
    """Learn wrong-source corrections while distilling correct source beliefs.
    Endpoint and teacher correctness control TRAIN loss only, never inference.
    """
    z=logits.float();teacher=teacher_logits.detach().float();logp=z.log_softmax(-1);p0=teacher.softmax(-1);log0=teacher.log_softmax(-1)
    world=bounds[:,None,:2]+grid_xy.float()*(bounds[:,None,2:]-bounds[:,None,:2]);dist=(world-target_world[:,None]).norm(dim=-1);inside=dist<=20;valid=inside.any(-1);outside=(~inside).any(-1);usable=valid&outside
    source_ids=teacher.argmax(-1);source_correct=dist.gather(1,source_ids[:,None])[:,0]<=20
    kl=(p0*(log0-logp)).sum(-1);mass=z.new_zeros(len(z));margin=z.new_zeros(len(z))
    if valid.any():mass[valid]=-torch.logsumexp(logp[valid].masked_fill(~inside[valid],-torch.inf),-1)
    if usable.any():
        pos=z[usable].masked_fill(~inside[usable],-torch.inf).max(-1).values;neg=z[usable].masked_fill(inside[usable],-torch.inf).max(-1).values;margin[usable]=F.relu(.5+neg-pos)
    anchor=z.gather(1,source_ids[:,None])[:,0];mask=torch.ones_like(z,dtype=torch.bool);mask.scatter_(1,source_ids[:,None],False);competitor=z.masked_fill(~mask,-torch.inf).max(-1).values
    preservation=kl+F.relu(competitor-anchor)
    corrective=mass+margin+.05*kl
    per_example=torch.where(source_correct,preservation,corrective)
    # Outside-grid endpoints still have a valid nearest-cell training target;
    # do not manufacture a <=20m positive for the region/ranking terms.
    absent=~valid
    if absent.any():
        nearest=dist[absent].argmin(-1);per_example[absent]=F.cross_entropy(z[absent],nearest,reduction='none')+.05*kl[absent]
    return per_example.mean(),dict(source_correct=int(source_correct.sum()),source_wrong=int((~source_correct).sum()),no_positive20=int(absent.sum()),fidelity_KL=float(kl.mean().detach()))
