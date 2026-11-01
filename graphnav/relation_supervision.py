"""Loss-only weak compass supervision. Target coordinates never enter the head."""
import torch
from torch.nn import functional as F


def compass_counterfactual_loss(real,opposite,world,target,known,pair_geometry,roles):
    legal=(world-target[:,None,:2]).norm(dim=-1)<=20
    index=roles['role_reference_index'].clamp_min(0)
    displacement=pair_geometry.float()[...,:2].gather(2,index[:,None,None,None].expand(-1,world.shape[1],1,2)).squeeze(2)
    projection=(displacement*roles['role_direction'][:,None]).sum(-1)
    # Require all legal cells to be safely on the claimed side, plus actual target RGB.
    aligned=projection.masked_fill(~legal,1e9).amin(-1)>.02
    eligible=roles['role_valid'] & (roles['role_direction'].abs().sum(-1)>0) & aligned & (legal&known).any(-1)
    weights=(legal&known).float();weights=weights/weights.sum(-1,keepdim=True).clamp_min(1)
    preference=((real-opposite)*weights).sum(-1)
    loss=(F.softplus((.05-preference)/.1)*.1*eligible).sum()/eligible.sum().clamp_min(1)
    return loss,eligible
