"""Observed-local position hypotheses compete with frozen global hypotheses.

This is an opt-in experiment, NOT an edit to original Stage1. Private coordinates
exist only in position_loss. Missing observation preserves the global posterior.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


class LocalChainPositionHead(nn.Module):
    def __init__(self, dimension=1024, hidden=64):
        super().__init__()
        self.image=nn.Linear(dimension,hidden)
        self.text=nn.Linear(dimension,hidden)
        self.geometry=nn.Linear(31,hidden)
        self.score=nn.Sequential(nn.Linear(hidden*4+9,hidden),nn.GELU(),nn.Linear(hidden,1))
        self.local_mass_bias=nn.Parameter(torch.tensor(-3.))

    def forward(self,*,prior_logits,local_uv,image_features,target_text,context_text,
                public_geometry,quality,local_valid):
        if prior_logits.ndim!=2 or int(math.isqrt(prior_logits.shape[1]))**2!=prior_logits.shape[1]:
            raise ValueError('square original global field required')
        batch,count,scales,dim=image_features.shape
        if scales!=3 or local_valid.dtype!=torch.bool or local_valid.shape!=(batch,count):
            raise ValueError('three observed scales and explicit validity required')
        if local_uv.shape!=(batch,count,2) or public_geometry.shape!=(batch,count,31) or quality.shape!=(batch,count,6):
            raise ValueError('observed candidate geometry/quality mismatch')
        if target_text.shape!=(batch,dim) or context_text.shape!=(batch,dim):
            raise ValueError('separate target/context text in shared visual space required')
        if not all(torch.isfinite(x).all() for x in (prior_logits,local_uv,image_features,target_text,context_text,public_geometry,quality)):
            raise ValueError('nonfinite public inputs')
        base=prior_logits.detach().float().log_softmax(-1)
        grid=int(math.isqrt(base.shape[-1]))
        sampled=F.grid_sample(base.exp().view(batch,1,grid,grid),
                              (local_uv.float()*2-1)[:,None],align_corners=False,padding_mode='border').reshape(batch,count)
        local_prior=sampled.clamp_min(1e-12).log()
        visual=F.normalize(image_features.float(),dim=-1)
        target=F.normalize(target_text.float(),dim=-1)
        context=F.normalize(context_text.float(),dim=-1)
        v=F.gelu(self.image(visual));t=F.gelu(self.text(target))[:,None]
        r=F.gelu(self.text(context))[:,None]
        native_target=(visual[:,:,0]*target[:,None]).sum(-1,keepdim=True)
        native_context=(visual[:,:,2]*context[:,None]).sum(-1,keepdim=True)
        combined=torch.cat((v[:,:,0]*t,v[:,:,1]*t,v[:,:,2]*r,
                            F.gelu(self.geometry(public_geometry.float())),quality.float(),
                            native_target,native_context,local_prior[...,None]),-1)
        learned=self.score(combined).squeeze(-1)
        # Correct for number of hypotheses. This does not infer target absence.
        local=local_prior + learned + self.local_mass_bias + math.log(base.shape[-1]) - local_valid.sum(-1,keepdim=True).clamp_min(1).float().log()
        local=local.masked_fill(~local_valid,float('-inf'))
        return torch.cat((base,local),-1)


def position_loss(logits,candidate_world,target_world,*,global_cells,radius_m=20.,sigma_m=10.):
    """World-goal supervision, NOT instance-positive pixel annotations.

    Local supervision is omitted if no local hypothesis is near the target.
    No unobserved position is labelled as a negative visual detection.
    """
    distance=(candidate_world.float()-target_world[:,None,:2].float()).norm(dim=-1)
    positive=(distance<=radius_m)&torch.isfinite(logits)
    eligible=positive.any(-1)
    logprob=logits.log_softmax(-1)
    good_mass=torch.logsumexp(logprob.masked_fill(~positive,float('-inf')),-1)
    goal=-good_mass[eligible].mean() if eligible.any() else logits[:,:global_cells].sum()*0
    local_logits=logits[:,global_cells:]
    local_distance=distance[:,global_cells:]
    local_positive=positive[:,global_cells:]
    local_eligible=local_positive.any(-1)
    if local_eligible.any():
        local_target=torch.exp(-.5*(local_distance[local_eligible]/sigma_m).square())*local_positive[local_eligible]
        local_target=local_target/local_target.sum(-1,keepdim=True).clamp_min(1e-12)
        lp=local_logits[local_eligible].log_softmax(-1)
        local=-(local_target*lp.masked_fill(~torch.isfinite(lp),0)).sum(-1).mean()
    else:
        local=logits[:,:global_cells].sum()*0
    return dict(total=goal+.5*local,goal=goal,local=local,local_supervised=int(local_eligible.sum()),
                global_supervised=int(eligible.sum()))
