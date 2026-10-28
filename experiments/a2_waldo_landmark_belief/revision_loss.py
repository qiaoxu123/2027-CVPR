"""Offline batched spatial/benefit losses; GT is excluded from model inputs."""
import torch
from torch.nn import functional as F
from .model import supervision_loss
from .revision_fusion import batch_proposal,batch_belief_mix

LOSS_WEIGHTS=dict(proposal=.5,category=.2,map=.5,preserve=2.,benefit=.2,expected_ne_risk=.2)


def revision_loss(output,batch,*,refine=False):
    labels=batch['labels'];s=batch['spatial']
    original,stats=supervision_loss(output,labels,refine=False);zero=output['scores'].sum()*0
    valid_category=labels['target_kind']>=0
    category=F.cross_entropy(output['category_logits'][valid_category],labels['target_kind'][valid_category]) if valid_category.any() else zero
    q,available=batch_proposal(s,output['scores'],output['candidate_match_logits'],output['reliable'])
    base=s['base_logits'].float();grid=s['grid_world'].float();target=labels['target_xy'].float()
    distance=(grid-target[:,None]).norm(dim=-1)
    sigma=((grid[:,1]-grid[:,0]).norm(dim=-1)/2).clamp_min(4.)
    soft_target=(-.5*(distance/sigma[:,None]).square()).softmax(-1)
    support=(s['candidate_support'].bool() & output['reliable'][:,:,None]).any(1)
    compatible=support & (distance<=20);covered=available & compatible.any(-1)
    t=soft_target*compatible;t=t/t.sum(-1,keepdim=True).clamp_min(1e-30)
    proposal_rows=-(t*q.clamp_min(1e-30).log()).sum(-1)
    proposal=proposal_rows[covered].mean() if covered.any() else zero
    base_logp=base.log_softmax(-1);base_p=base_logp.exp()
    mixed=batch_belief_mix(base,q,output['gate']);mixed_logp=mixed.log_softmax(-1);mixed_p=mixed_logp.exp()
    before=distance.gather(1,base.argmax(-1,keepdim=True)).squeeze(-1)
    after=distance.gather(1,mixed.argmax(-1,keepdim=True)).squeeze(-1)
    proposal_ne=distance.gather(1,q.argmax(-1,keepdim=True)).squeeze(-1)
    before_correct=before<=20;after_correct=after<=20
    map_loss=-(soft_target*mixed_logp).sum(-1).mean()
    base_expected=(base_p*distance).sum(-1);mixed_expected=(mixed_p*distance).sum(-1)
    ne_risk=(F.relu(mixed_expected-base_expected)/20).mean()
    preservation=(base_p*(base_logp-mixed_logp)).sum(-1).clamp_min(0)
    risk=preservation[before_correct].mean() if before_correct.any() else zero
    helpful=available & ((q.detach()*distance).sum(-1)+1.<base_expected.detach())
    benefit=F.binary_cross_entropy_with_logits(output['benefit_logit'],helpful.float())
    loss=original+LOSS_WEIGHTS['proposal']*proposal+LOSS_WEIGHTS['category']*category
    if refine:loss=loss+LOSS_WEIGHTS['map']*map_loss+LOSS_WEIGHTS['preserve']*risk+LOSS_WEIGHTS['benefit']*benefit+LOSS_WEIGHTS['expected_ne_risk']*ne_risk
    values=dict(proposal_localization=proposal,category=category,map_loss=map_loss,preserve=risk,benefit=benefit,
                expected_ne_risk=ne_risk,gate_mean=output['gate'].mean(),proposal_covered_views=covered.sum(),
                proposal_available_views=available.sum(),proposal_correct=((proposal_ne<=20)&available).sum(),
                a0_correct=before_correct.sum(),map_correct=after_correct.sum(),
                corrected=(after_correct & ~before_correct).sum(),broken=(before_correct & ~after_correct).sum(),
                a0_ne_sum=before.sum(),map_ne_sum=after.sum(),proposal_ne_sum=proposal_ne[available].sum())
    # One host transfer per batch instead of per observation/per scalar.
    numbers=torch.stack([v.float() for v in values.values()]).detach().cpu().tolist()
    return loss,dict(stats,**dict(zip(values,numbers)))
