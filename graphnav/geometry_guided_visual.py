"""Visual scoring cannot read polygons, world coordinates or geometry logits."""
import torch
from torch import nn
from torch.nn import functional as F
from .world_likelihood_evidence import world_evidence_logits


class LandmarkVisualBelief(nn.Module):
    def __init__(self,dimension=1024,hidden=64):
        super().__init__()
        self.image=nn.Linear(dimension,hidden)
        self.text=nn.Linear(dimension,hidden)
        self.score=nn.Sequential(nn.Linear(hidden*2+2,hidden),nn.GELU(),nn.Linear(hidden,1))
        nn.init.zeros_(self.score[-1].weight);nn.init.zeros_(self.score[-1].bias)

    def forward(self,*,image_features,name_text,type_text,valid,return_embedding=False):
        image=F.normalize(image_features[:,:,0].float(),dim=-1)
        name=F.normalize(name_text.float(),dim=-1);typ=F.normalize(type_text.float(),dim=-1)
        native_name=(image*name[:,None]).sum(-1)
        native_type=(image*typ[:,None]).sum(-1)
        native=10*(.25*native_name+.75*native_type)
        vi=F.gelu(self.image(image));vn=F.gelu(self.text(name))[:,None];vt=F.gelu(self.text(typ))[:,None]
        adaptation=self.score(torch.cat((vi*vn,vi*vt,native_name[...,None],native_type[...,None]),-1)).squeeze(-1)
        score=(native+adaptation).masked_fill(~valid,0)
        return (score,F.normalize(vi,dim=-1)) if return_embedding else score


def visual_world_beliefs(geometry_logits,local_uv,scores,valid,projection_supported):
    """Observed conditional V and global G+V have explicitly different domains."""
    uniform=torch.zeros_like(geometry_logits)
    uniform_lp=uniform.float().log_softmax(-1)
    energy=world_evidence_logits(uniform,local_uv,scores,valid)-uniform_lp
    support=world_evidence_logits(uniform,local_uv,torch.ones_like(scores),valid)-uniform_lp
    covered=support>.5
    raw=(uniform_lp+energy).masked_fill(~covered,float('-inf'))
    # Empty observation -> abstain; placeholder values never count as raw success.
    has_view=covered.any(-1)
    raw=torch.where(has_view[:,None],raw,uniform_lp)
    fused=geometry_logits.detach().float().log_softmax(-1)+energy*projection_supported[:,None]
    return dict(raw=raw,fused=fused,covered=covered,has_view=has_view,energy=energy)


def landmark_visual_loss(model,batch):
    score,embedding=model(image_features=batch['image_features'],name_text=batch['name_text'],
        type_text=batch['type_text'],valid=batch['valid'],return_embedding=True)
    labels=batch['labels'];valid=batch['valid'];positive=labels==1;negative=labels==0
    region=(F.softplus(-score)*positive).sum(-1)/positive.sum(-1).clamp_min(1)
    region=region+(F.softplus(score)*negative).sum(-1)/negative.sum(-1).clamp_min(1)
    region=region.mean()
    teacher=F.grid_sample(batch['geometry_logits'].float().view(-1,1,28,28),(batch['local_uv'].float()*2-1)[:,None],
        align_corners=False,padding_mode='border').squeeze(1).squeeze(1)
    target=teacher.masked_fill(~valid,float('-inf')).softmax(-1).detach()
    logprob=score.masked_fill(~valid,float('-inf')).log_softmax(-1).masked_fill(~valid,0)
    entropy=target.clamp_min(1e-12).log()
    distill=(target*(entropy-logprob)).sum(-1).mean()
    pooled=F.normalize((embedding*positive[...,None]).sum(1)/positive.sum(-1,keepdim=True).clamp_min(1),dim=-1)
    similarity=pooled@pooled.T/.1
    identity=torch.eye(len(pooled),device=score.device,dtype=torch.bool)
    group=batch['group_ids'];same=(group[:,None]==group[None])&~identity
    same=same&(batch['view_ids'][:,None]!=batch['view_ids'][None])
    eligible=same.any(-1)
    contrast=score.sum()*0
    if eligible.any():
        selected=similarity[eligible]
        denom=torch.logsumexp(selected.masked_fill(identity[eligible],float('-inf')),-1)
        numerator=torch.logsumexp(selected.masked_fill(~same[eligible],float('-inf')),-1)
        contrast=(denom-numerator).mean()
    return dict(total=region+.25*distill+.1*contrast,region=region,distill=distill,contrast=contrast,
        contrast_rows=int(eligible.sum()))
