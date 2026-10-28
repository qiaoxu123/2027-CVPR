"""Public-only FP32 proposal consistency envelope, not cross-device equality.

Actual captured GPU Q/action remain authoritative. World-coordinate cdist uses
norm/cross-product arithmetic in FP32; cancellation error scales with absolute
coordinate magnitudes, not just the small local distance. A conservative
32-roundoff screening envelope is propagated through normalized kernels and
weights. This is an audit under ordinary FP32 arithmetic, not a claim of
bitwise CPU/GPU identity or a hardware-independent formal error certificate.
"""
import torch
from torch.nn import functional as F


def audit_gpu_proposal(spatial,prediction,q,available,*,sigma=4.):
    u=torch.finfo(torch.float32).eps/2
    def gamma(n):return n*u/(1-n*u)
    scores=prediction['scores'].double();trust=prediction['candidate_match_logits'].double()
    support=spatial['candidate_support'].bool();valid=prediction['reliable'].bool()&support.any(-1)
    q=q.double();grid=spatial['grid_world'].double();xy=spatial['candidate_xy'].double()
    if q.shape!=(len(grid),) or support.shape!=(len(scores),len(grid)) or not torch.isfinite(q).all() or (q<0).any():raise ValueError('Invalid actual GPU probability/support shape')
    if bool(valid.any())!=bool(available):raise ValueError('GPU proposal availability differs from public support')
    if not available:
        if q.any():raise ValueError('No-evidence GPU proposal must be exactly zero')
        return dict(passed=True,max_abs_double_reference_difference=0.,maximum_envelope_width=0.)
    support=support[valid];xy=xy[valid];scores=scores[valid];trust=trust[valid]
    union=support.any(0)
    if (q[~union]!=0).any() or abs(float(q.sum())-1)>gamma(4*(len(scores)+len(grid))+64):raise ValueError('GPU proposal violates reliable support/normalization')
    def intervals(logits,error,mask):
        center=logits.masked_fill(~mask,-torch.inf).max(-1,keepdim=True).values
        low=(logits-error-center).masked_fill(~mask,-torch.inf).exp()
        high=(logits+error-center).masked_fill(~mask,-torch.inf).exp()
        lower=low/(low+(high.sum(-1,keepdim=True)-high).clamp_min(0)).clamp_min(1e-300)
        upper=high/(high+(low.sum(-1,keepdim=True)-low).clamp_min(0)).clamp_min(1e-300)
        return lower,upper
    d2=(xy[:,None]-grid[None]).square().sum(-1)
    kernel=-d2/(2*sigma*sigma)
    magnitude=xy.square().sum(-1)[:,None]+grid.square().sum(-1)[None]+2*(xy[:,None]*grid[None]).abs().sum(-1)
    error=gamma(32)*(magnitude+d2+1)/(2*sigma*sigma)+8*u*(1+kernel.abs())
    kl,ku=intervals(kernel,error,support)
    logits=scores+F.logsigmoid(trust);we=gamma(32)*(1+scores.abs()+trust.abs())
    wl,wu=intervals(logits[None],we[None],torch.ones_like(logits[None],dtype=torch.bool));wl=wl[0];wu=wu[0]
    lower=(wl[:,None]*kl).sum(0);upper=(wu[:,None]*ku).sum(0)
    lower=(lower*(1-gamma(4*len(scores)+32))-8*u).clamp_min(0)
    upper=upper*(1+gamma(4*len(scores)+32))+8*u
    if not torch.isfinite(lower).all() or not torch.isfinite(upper).all() or ((q<lower)|(q>upper)).any():raise ValueError('GPU proposal outside public FP32 numerical envelope')
    reference=(logits.softmax(-1)[:,None]*kernel.masked_fill(~support,-torch.inf).softmax(-1)).sum(0)
    return dict(passed=True,max_abs_double_reference_difference=float((q-reference).abs().max()),maximum_envelope_width=float((upper-lower).max()))
