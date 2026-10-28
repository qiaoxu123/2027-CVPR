"""Private spatial-utility losses, separate from public evidence/risk inference.

Spatial negatives mean unsuitable kernel locations for this TRAIN target;
they never assert a detected object's identity. Every per-row objective uses
an absolute source multiplier, so real-state weight does not cancel in a batch.
"""
import torch
from torch.nn import functional as F
from .relation_v3_fusion import proposal_with_quality
from .revision_fusion import batch_belief_mix
from .risk_probe import public_risk_features, FEATURE_NAMES

FEATURE_NAMES_V3=FEATURE_NAMES[:13]+('evidence_match_quality','proposal_available')
LOSS_WEIGHTS=dict(direction=.2,distance=.2,trust=.2,category=.2,proposal=.5,
                  map=.5,preserve=2.,expected_ne=.2,risk=.2)
FIXED_ACTION_WEIGHT=.25


def relation_v3_loss(model,output,batch,*,refine=False):
    labels=batch['labels'];s=batch['spatial'];scores=output['scores'].float()
    source_weight=labels['sample_weight'].float()
    if source_weight.shape!=(len(scores),) or not torch.isfinite(source_weight).all() or (source_weight<=0).any():
        raise ValueError('Invalid per-row training weights')
    def average(rows):return (rows*source_weight).mean()
    eligible=labels['utility_valid'].bool()&output['reliable'].bool()&s['candidate_support'].bool().any(-1)
    positive=labels['localization_utility'].bool()&eligible
    contrast=positive.any(-1)&(eligible&~positive).any(-1)
    safe_scores=torch.where(eligible.any(-1,keepdim=True),scores.masked_fill(~eligible,-torch.inf),torch.zeros_like(scores))
    rank_target=positive.float()/positive.sum(-1,keepdim=True).clamp_min(1)
    # Mask before multiplying: zero * -inf would produce NaN.
    rank_rows=-(rank_target*safe_scores.log_softmax(-1).masked_fill(~positive,0)).sum(-1)*contrast
    trust_rows=F.binary_cross_entropy_with_logits(output['candidate_match_logits'].float(),positive.float(),reduction='none')
    trust_rows=(trust_rows*eligible).sum(-1)/eligible.sum(-1).clamp_min(1)
    refs=labels['angle_valid'].bool()&output['reference_valid'].bool()
    direction_rows=((1-(output['angle']*labels['angle']).sum(-1))*refs).sum(-1)/refs.sum(-1).clamp_min(1)
    distance_rows=(F.smooth_l1_loss(output['distance'],labels['distance'],reduction='none')*refs).sum(-1)/refs.sum(-1).clamp_min(1)
    kind=labels['target_kind'].long();category_rows=F.cross_entropy(output['category_logits'],kind,ignore_index=-1,reduction='none')
    q,available,quality=proposal_with_quality(s,scores,output['candidate_match_logits'],output['reliable'])
    base=s['base_logits'].float();grid=s['grid_world'].float();target=labels['target_xy'].float()
    distance=(grid-target[:,None]).norm(dim=-1)
    # Supervise only target-near support of spatially useful, compatible boxes.
    useful_support=(s['candidate_support'].bool()&positive[:,:,None]).any(1)&(distance<=20)
    sigma=((grid[:,1]-grid[:,0]).norm(dim=-1)/2).clamp_min(4.)
    soft_target=(-.5*(distance/sigma[:,None]).square()).softmax(-1)
    proposal_target=soft_target*useful_support
    proposal_target=proposal_target/proposal_target.sum(-1,keepdim=True).clamp_min(1e-30)
    proposal_rows=-(proposal_target*q.clamp_min(1e-30).log()).sum(-1)
    covered=available&useful_support.any(-1)
    proposal_rows=proposal_rows*covered
    terms=dict(ranking=average(rank_rows),trust=average(trust_rows),direction=average(direction_rows),
               distance=average(distance_rows),category=average(category_rows),proposal=average(proposal_rows))
    loss=terms['ranking']+sum(LOSS_WEIGHTS[k]*terms[k] for k in ('trust','direction','distance','category','proposal'))
    mixed=batch_belief_mix(base,q,torch.full_like(quality,FIXED_ACTION_WEIGHT))
    base_logp=base.log_softmax(-1);base_p=base_logp.exp();mixed_logp=mixed.log_softmax(-1);mixed_p=mixed_logp.exp()
    before=distance.gather(1,base.argmax(-1,keepdim=True)).squeeze(-1)
    after=distance.gather(1,mixed.argmax(-1,keepdim=True)).squeeze(-1)
    correct_before=before<=20;correct_after=after<=20
    gain=~correct_before&correct_after;harm=correct_before&~correct_after
    zero=scores.sum()*0
    terms.update(map=zero,preserve=zero,expected_ne=zero,risk=zero)
    if refine:
        terms['map']=average(-(soft_target*mixed_logp).sum(-1)*available)
        terms['preserve']=average((base_p*(base_logp-mixed_logp)).sum(-1).clamp_min(0)*correct_before*available)
        terms['expected_ne']=average(F.relu(((mixed_p-base_p)*distance).sum(-1))/20*available)
        # Only public summaries enter the head, detached by predict_risk.
        features=public_risk_features(base,q,grid,quality,available)
        logits=model.predict_risk(features)
        outcomes=(gain.long()+2*harm.long()).detach()
        risk_rows=F.cross_entropy(logits,outcomes,weight=model.risk_class_weights,reduction='none')
        terms['risk']=average(risk_rows)
        loss=loss+sum(LOSS_WEIGHTS[k]*terms[k] for k in ('map','preserve','expected_ne','risk'))
    values=dict(terms,rank_views=contrast.sum(),utility_candidates=eligible.sum(),positive_candidates=positive.sum(),
        proposal_available_views=available.sum(),proposal_covered_views=covered.sum(),
        a0_correct=correct_before.sum(),map_correct=correct_after.sum(),corrected=gain.sum(),broken=harm.sum(),
        a0_ne_sum=before.sum(),map_ne_sum=after.sum())
    numbers=torch.stack([v.float() for v in values.values()]).detach().cpu().tolist()
    return loss,dict(zip(values,numbers))
