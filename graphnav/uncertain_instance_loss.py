"""Soft TRAIN proxy preferences, never a verified pixel-instance likelihood."""
import torch
from torch.nn import functional as F


def uncertain_instance_loss(scores,labels,valid,preference):
    if scores.shape!=labels.shape or valid.shape!=scores.shape or preference.shape!=(len(scores),):
        raise ValueError('aligned observed scores/private labels/preferences required')
    if not torch.isfinite(scores).all() or not ((labels>=-1)&(labels<=1)).all():raise ValueError('finite scores and ternary proxy labels required')
    if not torch.isfinite(preference).all() or not ((preference>=.5)&(preference<1)).all():raise ValueError('soft preferences in [0.5,1) required')
    positive=(labels==1)&valid;negative=(labels==0)&valid
    use=positive.any(-1)&negative.any(-1)&(preference>.5)
    if not use.any():return scores.sum()*0,0
    pos=torch.logsumexp(scores[use].masked_fill(~positive[use],float('-inf')),-1)-positive[use].sum(-1).float().log()
    neg=torch.logsumexp(scores[use].masked_fill(~negative[use],float('-inf')),-1)-negative[use].sum(-1).float().log()
    # One-sided finite-margin preference: do not erase a stronger existing
    # ordering merely because a weak teacher has limited confidence.
    margin=torch.logit(preference[use])
    return F.relu(margin-(pos-neg)).mean(),int(use.sum())
