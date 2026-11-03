"""Annotation-side weak instance ranking; never called by inference."""
import math
import torch
from torch.nn import functional as F


def instance_ranking_loss(scores,labels,valid,margin=.2):
    if scores.shape!=labels.shape or valid.shape!=scores.shape or valid.dtype!=torch.bool:
        raise ValueError('aligned scores, private instance labels and bool validity required')
    if not torch.isfinite(scores).all() or not math.isfinite(margin):raise ValueError('finite scores/margin required')
    if not ((labels==-1)|(labels==0)|(labels==1)).all():raise ValueError('instance labels must be unknown/other/target')
    positive=(labels==1)&valid;negative=(labels==0)&valid;eligible=positive.any(-1)&negative.any(-1)
    if not eligible.any():return scores.sum()*0,dict(rows=0,positives=0,negatives=0)
    values=scores[eligible].float();pos=positive[eligible];neg=negative[eligible]
    pos_mean=torch.logsumexp(values.masked_fill(~pos,float('-inf')),-1)-pos.sum(-1).float().log()
    neg_mean=torch.logsumexp(values.masked_fill(~neg,float('-inf')),-1)-neg.sum(-1).float().log()
    return F.softplus(margin+neg_mean-pos_mean).mean(),dict(rows=int(eligible.sum()),positives=int(pos.sum()),negatives=int(neg.sum()))


def noncar_position_auxiliary(scores,distance,valid,categories):
    """Preserve old non-Car row weighting, omit coarse Car pseudo positives."""
    positive=valid&(distance<=20);eligible=positive.any(-1)
    noncar=torch.tensor([c!='Car' for c in categories],device=scores.device)
    use=eligible&noncar
    if not use.any():return scores.sum()*0
    weights=torch.exp(-.5*(distance[use]/10.).square())*positive[use]
    weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)
    lp=scores[use].masked_fill(~valid[use],float('-inf')).log_softmax(-1).masked_fill(~valid[use],0)
    return -(weights*lp).sum()/eligible.sum().clamp_min(1)
