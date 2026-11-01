"""Public native-query and belief summaries for supervised arrival verification."""
import math
import torch

FEATURE_NAMES=tuple([f'query_mass_{r}m' for r in (5,10,15,20,30,40)]+['visual_mass','max_query_probability','query_distance','query_spread','query_entropy','top_query_distance','near_quality','prior_mass20m','belief_mass20m','prior_goal_distance','belief_goal_distance','belief_entropy'])


def arrival_features(query_logits,null_logits,query_world,valid,geometry,base_logits,belief_logits,grid_world,pose):
    """Batched public tensors; target coordinates are intentionally not an argument."""
    scores=torch.cat((query_logits.float().masked_fill(~valid,-1e4),null_logits.float()[:,None]),-1)
    probability=scores.softmax(-1);p=probability[:,:-1]*valid
    distance=(query_world.float()-pose[:,None,:2]).norm(dim=-1);mass=p.sum(-1)
    conditional=p/mass[:,None].clamp_min(1e-8)
    mean=(conditional[...,None]*query_world).sum(1)
    spread=((query_world-mean[:,None]).norm(dim=-1)*conditional).sum(-1).clamp_max(100)/100
    values=[(p*(distance<=r)).sum(-1) for r in (5,10,15,20,30,40)]
    maximum,index=p.max(-1);top_distance=distance.gather(1,index[:,None])[:,0].clamp_max(100)/100
    entropy=-(probability*probability.clamp_min(1e-8).log()).sum(-1)/math.log(probability.shape[-1])
    near=p*(distance<=20);quality=(near*geometry[...,2]).sum(-1)/near.sum(-1).clamp_min(1e-8)
    values += [mass,maximum,(conditional*distance.clamp_max(100)).sum(-1)/100,spread,entropy,top_distance,quality]
    grid_distance=(grid_world-pose[:,None,:2]).norm(dim=-1)
    prior=base_logits.float().softmax(-1);belief=belief_logits.float().softmax(-1)
    values += [(prior*(grid_distance<=20)).sum(-1),(belief*(grid_distance<=20)).sum(-1),grid_distance.gather(1,base_logits.argmax(-1)[:,None])[:,0].clamp_max(100)/100,grid_distance.gather(1,belief_logits.argmax(-1)[:,None])[:,0].clamp_max(100)/100,-(belief*belief.clamp_min(1e-8).log()).sum(-1)/math.log(belief.shape[-1])]
    result=torch.stack(values,-1)
    if result.shape[-1]!=len(FEATURE_NAMES) or not torch.isfinite(result).all():raise ValueError('Invalid public arrival features')
    return result
