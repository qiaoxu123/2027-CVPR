"""Trainable native object-query evidence for a frozen global goal belief.

All forward inputs are public observations/geometric priors. Target coordinates
are used exclusively by object_evidence_loss, not query selection or forward.
"""
import argparse
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .aerialvg_pair_residual import official_relation_class
from .workspace_paths import WORKSPACE


class ObjectQueryRanker(nn.Module):
    def __init__(self,relation=False,geometry_dim=8,train_relation=False):
        super().__init__();rng=torch.random.get_rng_state();self.relation=None
        if relation:
            self.relation=official_relation_class()()
            with torch.serialization.safe_globals([argparse.Namespace]):s=torch.load(WORKSPACE/'third_party/aerialvg_assets/aerialvg.pth',map_location='cpu',weights_only=True)
            self.relation.load_state_dict({k.removeprefix('relation_transformer.'):v for k,v in s['model'].items() if k.startswith('relation_transformer.')},strict=True)
            self.relation.requires_grad_(train_relation)
        torch.random.set_rng_state(rng)
        self.score=nn.Sequential(nn.Linear(256*3+geometry_dim,128),nn.GELU(),nn.Linear(128,1))
        self.null=nn.Sequential(nn.Linear(256+geometry_dim,64),nn.GELU(),nn.Linear(64,1))
        nn.init.zeros_(self.score[-1].weight);nn.init.zeros_(self.score[-1].bias)
        nn.init.zeros_(self.null[-1].weight);nn.init.constant_(self.null[-1].bias,4.)
        self.fusion_gain=nn.Parameter(torch.zeros(()))

    def forward(self,features,text,text_mask,geometry,valid):
        language=(text*text_mask[...,None]).sum(1)/text_mask.sum(1,keepdim=True).clamp_min(1)
        token_logits=features@text.transpose(-1,-2)
        native=token_logits.masked_fill(~text_mask[:,None],-1e4).amax(-1).float()
        if self.relation is not None:
            # Upstream has no candidate-padding mask; remove padding before forming K² pairs.
            if features.shape[1]>15:raise ValueError('Select at most15 queries before official pair attention')
            native=native.clone()
            for b in range(len(features)):
                ids=valid[b].nonzero().flatten()
                if not len(ids) or not text_mask[b].any():continue
                pair=self.relation(features[b:b+1,ids],dict(encoded_text=text[b:b+1],text_token_mask=text_mask[b:b+1]))[0,:,:text.shape[1]]
                combined=.6*token_logits[b,ids]+.4*pair
                native[b,ids]=combined.masked_fill(~text_mask[b][None],-1e4).amax(-1)
        language=language[:,None].expand_as(features)
        query=torch.cat((features,language,features*language,geometry),-1)
        logits=native+2*self.score(query).squeeze(-1).tanh()
        logits=logits.masked_fill(~valid,-1e4)
        weights=valid.float()/valid.sum(-1,keepdim=True).clamp_min(1)
        summary=torch.cat(((language*weights[...,None]).sum(1),(geometry*weights[...,None]).sum(1)),-1)
        null=self.null(summary).squeeze(-1)
        return logits,null

    def update_belief(self,base_logits,query_logits,null,kernel,observed,valid,bound=2.):
        query_logits=query_logits.float().masked_fill(~valid,-1e4)
        probability=torch.cat((query_logits,null.float()[:,None]),-1).softmax(-1)
        support=kernel.float()*observed[...,None]*valid[:,None]
        support=support/support.sum(1,keepdim=True).clamp_min(1e-9)
        density=(support*probability[:,:-1,None].transpose(1,2)).sum(-1)
        evidence_mass=probability[:,:-1].sum(-1,keepdim=True)
        density=density/evidence_mass.clamp_min(1e-9)
        prior=base_logits.float().softmax(-1)*observed
        prior=prior/prior.sum(-1,keepdim=True).clamp_min(1e-9)
        ratio=(density.clamp_min(1e-9).log()-prior.clamp_min(1e-9).log()).clamp(-bound,bound)
        available=(support.sum((1,2))>0)[:,None]
        delta=self.fusion_gain.clamp_min(0).tanh()*evidence_mass*ratio*observed*available
        return base_logits.float()+delta,delta


def object_evidence_loss(query_logits,null,query_world,target_world,valid,updated_logits,grid_world,base_logits):
    legal_query=((query_world-target_world[:,None,:2]).norm(dim=-1)<=20)&valid
    scores=torch.cat((query_logits.float().masked_fill(~valid,-1e4),null.float()[:,None]),-1)
    positives=torch.cat((legal_query,~legal_query.any(-1,keepdim=True)),-1)
    classification=(torch.logsumexp(scores,-1)-torch.logsumexp(scores.masked_fill(~positives,-1e4),-1)).mean()
    legal_grid=(grid_world-target_world[:,None,:2]).norm(dim=-1)<=20
    nearest=(grid_world-target_world[:,None,:2]).norm(dim=-1).argmin(-1)
    legal_grid=legal_grid.clone();legal_grid[torch.arange(len(legal_grid),device=legal_grid.device),nearest]|=~legal_grid.any(-1)
    localization=(torch.logsumexp(updated_logits,-1)-torch.logsumexp(updated_logits.masked_fill(~legal_grid,-1e4),-1)).mean()
    drift=F.kl_div(updated_logits.log_softmax(-1),base_logits.float().softmax(-1),reduction='batchmean')
    return dict(total=localization+.25*classification+.05*drift,localization=localization,classification=classification,drift=drift)
