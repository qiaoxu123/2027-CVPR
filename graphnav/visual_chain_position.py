"""Separate RGB-language and geometry energies; GT exists only in loss below."""
import math
import re
import torch
from torch import nn
from torch.nn import functional as F
from .local_chain_raster import rasterized_goal_logprob


class VisualChainPositionHead(nn.Module):
    grid_output=True
    def __init__(self,dimension=1024,hidden=64):
        super().__init__()
        self.image=nn.Linear(dimension,hidden);self.text=nn.Linear(dimension,hidden);self.geometry=nn.Linear(31,hidden)
        self.visual_score=nn.Sequential(nn.Linear(hidden*3+2,hidden),nn.GELU(),nn.Linear(hidden,1))
        self.spatial_score=nn.Sequential(nn.Linear(hidden+7,hidden),nn.GELU(),nn.Linear(hidden,1))
        self.local_mass_bias=nn.Parameter(torch.tensor(-3.))

    def forward(self,*,prior_logits,local_uv,image_features,target_text,context_text,public_geometry,quality,local_valid,return_visual=False):
        batch,count,scales,dimension=image_features.shape;cells=prior_logits.shape[-1];grid=math.isqrt(cells)
        if grid*grid!=cells or scales!=3 or local_valid.dtype!=torch.bool:raise ValueError('square global field and three observed scales required')
        base=prior_logits.detach().float().log_softmax(-1)
        local_prior=F.grid_sample(base.exp().view(batch,1,grid,grid),(local_uv.float()*2-1)[:,None],align_corners=False,padding_mode='border').reshape(batch,count).clamp_min(1e-12).log()
        image=F.normalize(image_features.float(),dim=-1);target=F.normalize(target_text.float(),dim=-1);context=F.normalize(context_text.float(),dim=-1)
        v=F.gelu(self.image(image));t=F.gelu(self.text(target))[:,None];r=F.gelu(self.text(context))[:,None]
        visual=self.visual_score(torch.cat((v[:,:,0]*t,v[:,:,1]*t,v[:,:,2]*r,
            (image[:,:,0]*target[:,None]).sum(-1,keepdim=True),(image[:,:,2]*context[:,None]).sum(-1,keepdim=True)),-1)).squeeze(-1)
        spatial=self.spatial_score(torch.cat((F.gelu(self.geometry(public_geometry.float()))*t,quality.float(),local_prior[...,None]),-1)).squeeze(-1)
        local=local_prior+visual+spatial+self.local_mass_bias+math.log(cells)-local_valid.sum(-1,keepdim=True).clamp_min(1).float().log()
        local=local.masked_fill(~local_valid,float('-inf'))
        joint=torch.cat((base,local),-1)
        return (joint,visual) if return_visual else joint


def visual_chain_loss(model,joint,visual,inputs,candidate_world,target_world,categories,phrases):
    cells=inputs['prior_logits'].shape[-1]
    grid_logits=rasterized_goal_logprob(joint,inputs['local_uv'],inputs['local_valid'],global_cells=cells)
    world_distance=(candidate_world.float()-target_world[:,None,:2]).norm(dim=-1)
    positive=world_distance[:,:cells]<=20.;eligible=positive.any(-1)
    good=torch.logsumexp(grid_logits.masked_fill(~positive,float('-inf')),-1)
    goal=-good[eligible].mean() if eligible.any() else visual.sum()*0
    pos_peak=grid_logits.masked_fill(~positive,float('-inf')).amax(-1)
    neg_peak=grid_logits.masked_fill(positive,float('-inf')).amax(-1)
    rank=F.relu(.2+neg_peak-pos_peak)[eligible].mean() if eligible.any() else visual.sum()*0
    local_distance=world_distance[:,cells:];local_positive=(local_distance<=20.)&inputs['local_valid']
    local_eligible=local_positive.any(-1)
    masked=visual.masked_fill(~inputs['local_valid'],float('-inf'))
    if local_eligible.any():
        weights=torch.exp(-.5*(local_distance[local_eligible]/10.).square())*local_positive[local_eligible]
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)
        lp=masked[local_eligible].log_softmax(-1).masked_fill(~inputs['local_valid'][local_eligible],0)
        visual_loss=-(weights*lp).sum(-1).mean()
    else:visual_loss=visual.sum()*0
    # Weak language-color alignment, NOT claimed pixel-instance supervision.
    # Positives share a color; never make identical gray cars false negatives.
    color=[]
    for category,phrase in zip(categories,phrases):
        words=set(re.findall(r'[a-z]+',phrase.lower().replace('grey','gray').replace('silver','gray')))
        colors=words&{'white','black','gray','red','blue','green','yellow','orange','brown'}
        color.append(next(iter(colors)) if category=='Car' and len(colors)==1 and words&{'car','van','truck','vehicle','bus','jeep','suv'} and not any(w.startswith(('roof','window')) for w in words) else None)
    nearest=local_distance.masked_fill(~inputs['local_valid'],float('inf')).amin(-1)
    selected=[i for i,c in enumerate(color) if c and nearest[i]<=5 and inputs['quality'][i,0,3]*100<=8]
    contrast=visual.sum()*0;contrast_rows=0
    if len({color[i] for i in selected})>=2:
        ids=torch.tensor(selected,device=visual.device)
        weights=torch.exp(-.5*(local_distance[ids]/2.).square())*inputs['local_valid'][ids]
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)
        projected=F.gelu(model.image(F.normalize(inputs['image_features'][ids,:,0].float(),dim=-1)))
        image=F.normalize((projected*weights[...,None]).sum(1),dim=-1)
        text=F.normalize(F.gelu(model.text(F.normalize(inputs['target_text'][ids].float(),dim=-1))),dim=-1)
        scores=image@text.T/.1
        same=torch.tensor([[color[i]==color[j] for j in selected] for i in selected],device=visual.device)
        contrast=(torch.logsumexp(scores,-1)-torch.logsumexp(scores.masked_fill(~same,float('-inf')),-1)).mean()
        contrast_rows=len(selected)
    return dict(total=goal+.1*rank+.5*visual_loss+.1*contrast,goal=goal,rank=rank,visual=visual_loss,
                contrast=contrast,local_supervised=int(local_eligible.sum()),color_rows=contrast_rows)


class WorldLikelihoodPositionHead(VisualChainPositionHead):
    """Same visual/geometry capacity; likelihood updates one fixed world support."""
    def __init__(self,dimension=1024,hidden=64):
        super().__init__(dimension,hidden)
        self.local_mass_bias.requires_grad_(False)
        for layer in (self.visual_score[-1],self.spatial_score[-1]):
            nn.init.normal_(layer.weight,std=1e-3);nn.init.zeros_(layer.bias)

    def forward(self,*,prior_logits,local_uv,image_features,target_text,context_text,public_geometry,quality,local_valid,return_visual=False):
        from .world_likelihood_evidence import world_evidence_logits
        joint,visual=super().forward(prior_logits=prior_logits,local_uv=local_uv,image_features=image_features,target_text=target_text,
            context_text=context_text,public_geometry=public_geometry,quality=quality,local_valid=local_valid,return_visual=True)
        cells=prior_logits.shape[-1];grid=math.isqrt(cells)
        prior=prior_logits.detach().float().log_softmax(-1)
        local_prior=F.grid_sample(prior.exp().view(-1,1,grid,grid),(local_uv.float()*2-1)[:,None],align_corners=False,padding_mode='border').reshape_as(local_valid).clamp_min(1e-12).log()
        energy=joint[:,cells:]-local_prior-self.local_mass_bias-math.log(cells)+local_valid.sum(-1,keepdim=True).clamp_min(1).float().log()
        updated=world_evidence_logits(prior_logits,local_uv,energy,local_valid)
        packed=torch.cat((updated,torch.full_like(energy,float('-inf'))),-1)
        return (packed,visual) if return_visual else packed
