"""Current reference appearance can inform global targets beyond camera coverage."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .evidence_reachability import local_evidence_coverage


class ReferenceEvidenceTransport(nn.Module):
    def __init__(self,dimension=1024,hidden=32):
        super().__init__()
        self.image=nn.Linear(dimension*2,hidden,bias=False)
        self.reference=nn.Linear(dimension,hidden,bias=False)
        self.relation=nn.Linear(dimension,hidden,bias=False)
        self.displacement=nn.Linear(6,hidden)
        self.kernel=nn.Linear(hidden,1,bias=False)
        self.evidence=nn.Linear(hidden,1,bias=False)
        self.log_scale=nn.Parameter(torch.tensor(math.log(50.)))
        nn.init.normal_(self.kernel.weight,std=.01)
        nn.init.zeros_(self.evidence.weight)

    def forward(self,*,global_world,local_world,image_features,reference_text,instruction_text,local_valid):
        if local_valid.dtype!=torch.bool or image_features.shape[:2]!=local_valid.shape or image_features.shape[-2]!=3:
            raise ValueError('aligned current-image scales and public validity required')
        image=F.normalize(image_features.float(),dim=-1)
        appearance=torch.cat((image[:,:,0],image[:,:,2]),-1)
        projected=F.layer_norm(torch.tanh(self.image(appearance)),(self.image.out_features,))
        reference=F.layer_norm(torch.tanh(self.reference(F.normalize(reference_text.float(),dim=-1))),(self.reference.out_features,))
        score=self.evidence(projected*reference[:,None]).squeeze(-1)
        delta=(global_world.float()[:,:,None]-local_world.float()[:,None])/100.
        distance=delta.norm(dim=-1,keepdim=True)
        geometry=torch.cat((delta,delta.abs(),distance,torch.log1p(distance)),-1)
        relation=torch.tanh(self.relation(F.normalize(instruction_text.float(),dim=-1)))[:,None,None]
        kernel=self.kernel(torch.tanh(self.displacement(geometry)*relation)).squeeze(-1)
        scale=self.log_scale.exp().clamp(5.,500.)/100.
        kernel=kernel-.5*(distance.squeeze(-1)/scale).square()
        kernel=kernel.masked_fill(~local_valid[:,None],float('-inf'))
        count=local_valid.sum(-1).clamp_min(1).float().log()[:,None,None]
        normalized=kernel-count
        background=torch.zeros(*normalized.shape[:2],1,device=normalized.device,dtype=normalized.dtype)
        # The always-finite neutral hypothesis avoids all-masked logsumexp NaNs.
        numerator=torch.logsumexp(torch.cat((background,normalized+score[:,None]),-1),-1)
        denominator=torch.logsumexp(torch.cat((background,normalized),-1),-1)
        return numerator-denominator


class ReferenceTransportPositionHead(nn.Module):
    grid_output=True
    def __init__(self,base,mode='global',dimension=1024,hidden=32):
        super().__init__()
        if mode not in ('global','local','constant'):raise ValueError('explicit matched transport mode required')
        self.base=base.eval().requires_grad_(False)
        self.transport=ReferenceEvidenceTransport(dimension,hidden)
        self.mode=mode
        self.register_buffer('constant_features',torch.zeros(3,dimension))

    def train(self,mode=True):
        super().train(mode);self.base.eval();return self

    def forward(self,*,prior_logits,local_uv,image_features,target_text,context_text,public_geometry,quality,local_valid,
                global_world,local_world,instruction_text,control='real',transport_image_features=None,return_evidence=False):
        with torch.no_grad():
            packed=self.base(prior_logits=prior_logits,local_uv=local_uv,image_features=image_features,target_text=target_text,
                context_text=context_text,public_geometry=public_geometry,quality=quality,local_valid=local_valid)
        valid=local_valid&(local_uv>=0).all(-1)&(local_uv<=1).all(-1)
        current=image_features if transport_image_features is None else transport_image_features
        if self.mode=='constant':current=self.constant_features[None,None].expand_as(current)
        if control=='zero-image':current=torch.zeros_like(current)
        elif control=='shuffle-image':
            current=current.clone()
            for i in range(len(current)):
                ids=valid[i].nonzero().flatten();current[i,ids]=current[i,ids.flip(0)].clone()
        elif control=='mean-image':
            mean=(current*valid[:,:,None,None]).sum(1)/valid.sum(-1).clamp_min(1)[:,None,None]
            current=mean[:,None].expand_as(current)
        elif control not in ('real','zero-relation'):raise ValueError('unknown transport-only diagnostic')
        instruction=instruction_text if control!='zero-relation' else torch.zeros_like(instruction_text)
        evidence=self.transport(global_world=global_world,local_world=local_world,image_features=current,
            reference_text=context_text,instruction_text=instruction,local_valid=valid)
        cells=prior_logits.shape[-1]
        if self.mode=='local':evidence=evidence*local_evidence_coverage(local_uv,valid,cells)
        result=torch.cat((packed[:,:cells]+evidence,packed[:,cells:]),-1)
        return (result,evidence) if return_evidence else result
