"""Public proposal/prior features and SEPARATE offline target outcome labels."""
import math
import torch
from .revision_fusion import batch_belief_mix

WEIGHTS=(.025,.0625,.125,.25)
FEATURE_NAMES=('prior_entropy','proposal_entropy','prior_peak','proposal_peak','prior_margin','proposal_margin',
               'agreement','js_divergence','expected_distance_from_prior_peak','peak_distance',
               'prior_mass_at_proposal_peak','proposal_mass_at_prior_peak','prior_mass_on_proposal_support',
               'old_benefit_probability','proposal_available')


def public_risk_features(base,q,grid,benefit,available):
    p=base.float().softmax(-1);q=q.float();grid=grid.float();epsilon=1e-30
    def entropy(x):return -(x*x.clamp_min(epsilon).log()).sum(-1)/math.log(max(x.shape[-1],2))
    def margin(x):
        top=x.topk(min(2,x.shape[-1]),-1).values
        return top[:,0]-top[:,-1]
    prior_peak=p.argmax(-1);proposal_peak=q.argmax(-1);batch=torch.arange(len(p),device=p.device)
    origin=grid[batch,prior_peak];distances=(grid-origin[:,None]).norm(dim=-1)/400.
    middle=.5*(p+q)
    js=.5*(p*(p.clamp_min(epsilon).log()-middle.clamp_min(epsilon).log())+q*(q.clamp_min(epsilon).log()-middle.clamp_min(epsilon).log())).sum(-1)
    result=torch.stack((entropy(p),entropy(q),p.max(-1).values,q.max(-1).values,margin(p),margin(q),
        (p*q).sum(-1),js,(q*distances).sum(-1),distances[batch,proposal_peak],
        p[batch,proposal_peak],q[batch,prior_peak],(p*(q>0)).sum(-1),benefit.float(),available.float()),-1)
    if not torch.isfinite(result).all():raise ValueError('Nonfinite public risk features')
    return result


def outcome_labels(base,q,grid,target):
    """PRIVATE offline supervision only; never passed to public feature/model code."""
    distance=(grid.float()-target.float()[:,None]).norm(dim=-1);batch=torch.arange(len(base),device=base.device)
    before=distance[batch,base.argmax(-1)];after=[]
    for weight in WEIGHTS:
        mixed=batch_belief_mix(base,q,torch.full((len(base),),weight,device=base.device))
        after.append(distance[batch,mixed.argmax(-1)])
    after=torch.stack(after,-1)
    gain=(before[:,None]>20)&(after<=20);harm=(before[:,None]<=20)&(after>20)
    return dict(outcomes=gain.to(torch.int8)-harm.to(torch.int8),before_ne=before,after_ne=after)
