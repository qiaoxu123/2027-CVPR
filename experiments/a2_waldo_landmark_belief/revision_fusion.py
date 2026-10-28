"""Label-free revision fusion: bounded GLOBAL mass transfer on observed support."""
import math
import torch
from torch.nn import functional as F

MAX_GATE=.25


def a0_candidate_context(spatial):
    """Six public per-box features derived from the actual current A0 field."""
    base=spatial['base_logits'].float();grid=spatial['grid_world'].float()
    support=spatial['candidate_support'].bool();xy=spatial['candidate_xy'].float()
    logp=base.log_softmax(-1);p=logp.exp();n=len(xy)
    inside=logp[None].expand(n,-1).masked_fill(~support,-torch.inf)
    mass=inside.logsumexp(-1).clamp(min=-30,max=0)/10
    gap=(inside.max(-1).values-logp.max()).clamp(min=-30,max=0)/10
    distance=(xy-grid[base.argmax()]).norm(dim=-1).clamp(max=400)/40
    entropy=-(p*logp).sum()/math.log(max(len(base),2))
    top=logp.topk(min(len(base),2)).values
    margin=(top[0]-top[-1]).clamp(max=20)/10
    size=support.float().mean(-1)
    return torch.stack((mass,gap,distance,entropy.expand(n),margin.expand(n),size),-1)


def proposal_distribution(spatial,scores,trust_logits,reliable,*,sigma_m=4.):
    """Rank/soft-trust normalized spatial proposal; no identity/target inputs."""
    if sigma_m<=0 or not math.isfinite(float(sigma_m)):raise ValueError('Invalid proposal sigma')
    support=spatial['candidate_support'].bool()
    if support.shape!=(len(scores),len(spatial['base_logits'])) or reliable.shape!=scores.shape:
        raise ValueError('Proposal tensor axes changed')
    valid=reliable.bool() & support.any(-1)
    if not valid.any():return torch.zeros_like(spatial['base_logits']),False
    xy=spatial['candidate_xy'][valid].float();grid=spatial['grid_world'].float()
    kernel=-.5*(torch.cdist(xy,grid)/sigma_m).square()
    kernel=kernel.masked_fill(~support[valid],-torch.inf).softmax(-1)
    weights=(scores[valid].float()+F.logsigmoid(trust_logits[valid].float())).softmax(0)
    return (weights[:,None]*kernel).sum(0),True


def bounded_belief_mix(base,proposal,gate):
    """Return original logit scale, with <=25% globally transferred mass.

    The proposal must be nonnegative. No support means exact fallback. Outside
    probabilities shrink by1-g; unlike old fusion, support mass can grow.
    """
    g=torch.as_tensor(gate,device=base.device,dtype=torch.float32)
    if g.ndim or not torch.isfinite(g) or not 0<=float(g.detach())<=MAX_GATE:
        raise ValueError('Revision gate must be a finite scalar in[0,.25]')
    if proposal.shape!=base.shape or not torch.isfinite(proposal).all() or (proposal<0).any():
        raise ValueError('Invalid spatial proposal')
    if float(g.detach())==0 or float(proposal.detach().sum())==0:return base
    q=proposal.float()/proposal.float().sum()
    logq=q.clamp_min(1e-38).log().masked_fill(q<=0,-torch.inf)
    normalizer=base.float().logsumexp(0)
    mixed=torch.logaddexp(base.float()-normalizer+torch.log1p(-g),logq+g.log())
    return mixed+normalizer


def batch_proposal(spatial,scores,trust_logits,reliable,*,sigma_m=4.):
    """Batched equivalent of proposal_distribution, including empty rows."""
    support=spatial['candidate_support'].bool();eligible=reliable.bool() & support.any(-1)
    available=eligible.any(-1)
    kernel=-.5*(torch.cdist(spatial['candidate_xy'].float(),spatial['grid_world'].float())/sigma_m).square()
    masked=kernel.masked_fill(~support,-torch.inf)
    safe=torch.where(support.any(-1,keepdim=True),masked,torch.zeros_like(masked))
    kernels=safe.softmax(-1)*support
    logits=(scores.float()+F.logsigmoid(trust_logits.float())).masked_fill(~eligible,-torch.inf)
    safe_logits=torch.where(available[:,None],logits,torch.zeros_like(logits))
    weights=safe_logits.softmax(-1)*eligible
    proposal=(weights[:,:,None]*kernels).sum(1)
    return proposal,available


def batch_belief_mix(base,proposal,gate):
    """Batched equivalent with exact zero-proposal/zero-gate fallback."""
    if not torch.isfinite(gate).all() or (gate<0).any() or (gate>MAX_GATE).any():raise ValueError('Invalid batched gate')
    mass=proposal.sum(-1,keepdim=True);q=proposal/mass.clamp_min(1e-30)
    active=(mass.squeeze(-1)>0) & (gate>0)
    g=torch.where(active,gate,torch.zeros_like(gate))
    logq=q.clamp_min(1e-38).log().masked_fill(q<=0,-torch.inf)
    normalizer=base.logsumexp(-1,keepdim=True)
    mixed=torch.logaddexp(base-normalizer+torch.log1p(-g[:,None]),logq+g[:,None].clamp_min(1e-38).log())+normalizer
    return torch.where(active[:,None],mixed,base)
