"""Versioned weak local-Anchor loss. Unknown pixels never become negatives."""
import torch
from torch.nn import functional as F
from .pgvb_training import pgvb_batch,frame_batches


def local_pgvb_loss(scores,labels,rows):
    positive=labels==1;negative=labels==0;known=positive|negative
    count=positive.sum(-1);negative_count=negative.sum(-1)
    if not (count>0).all():raise ValueError('No reliable positive; keep sample unknown')
    pos_lse=torch.logsumexp(scores.masked_fill(~positive,float('-inf')),-1)
    known_lse=torch.logsumexp(scores.masked_fill(~known,float('-inf')),-1)
    # Zero negatives: only balance the distribution WITHIN reliable positives.
    # Neither normalization nor any contrast includes unknown windows.
    uniform_positive_kl=pos_lse-(scores*positive).sum(-1)/count-count.float().log()
    region=torch.where(negative_count>0,known_lse-pos_lse,uniform_positive_kl).mean()
    bce=((F.softplus(-scores)*positive).sum(-1)/count+
        (F.softplus(scores)*negative).sum(-1)/negative_count.clamp_min(1)).mean()
    cpu=labels.detach().cpu();pos=cpu==1;neg=cpu==0;pairs=[]
    for i,a in enumerate(rows):
        for j,b in enumerate(rows):
            if i==j or a['frame_id']!=b['frame_id'] or a['type']!=b['type']:continue
            if (pos[j]&~neg[i]).any() or (pos[i]&~neg[j]).any():continue
            pairs.append(F.softplus(1+scores[i][positive[j]].mean()-scores[i][positive[i]].mean()))
    pair=torch.stack(pairs).mean() if pairs else scores.sum()*0
    return dict(total=region+.2*bce+.3*pair,region=region,bce=bce,pair=pair,pairs=len(pairs),
        positive_only_rows=int((negative_count==0).sum()))
