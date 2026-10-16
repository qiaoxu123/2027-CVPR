"""Observed RGB evidence correction conditioned on a frozen geometric belief.

Labels are accepted only by correction_loss, never by forward.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .full_localization import GoalField
from .stage1 import preserved_proposal_order


class CalibratedVisualResidual(nn.Module):
    def __init__(self,context_dim=128,vision_dim=1024,hidden=64,bound=2.,centered=True,use_pose=True):
        super().__init__();self.bound=float(bound);self.centered=bool(centered);self.use_pose=bool(use_pose)
        self.image=nn.Linear(vision_dim,32,bias=False)
        self.text=nn.Linear(vision_dim,32,bias=False)
        self.geometry=nn.Linear(context_dim,16)
        self.network=nn.Sequential(nn.Linear(16+32*2+18+8,hidden),nn.GELU(),nn.Linear(hidden,2))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
        with torch.no_grad():self.network[-1].bias[1]=-1.

    def forward(self,base,public,static_inputs):
        images=public['roi_features'].float();past=public['history_features'].float()
        valid=public['roi_valid'];remembered=public['history_valid']
        known=valid.any(-1)|remembered.any(-1)
        meaningful=(images.abs().sum((-1,-2))>0)|(past.abs().sum((-1,-2))>0)
        known=known&meaningful
        def pool(values,mask,quality):
            weight=(quality[...,0].clamp(0,1)*quality[...,1].clamp(0,1))*mask
            weight=weight/weight.sum(-1,keepdim=True).clamp_min(1e-9)
            return (values*weight[...,None]).sum(-2)
        current=pool(images,valid,public['roi_quality']);history=pool(past,remembered,public['history_quality'])
        language=F.normalize(public['native_text'].float()+public['appearance_text'].float(),dim=-1)
        instruction=self.text(language)[:,None]
        rgb_current=self.image(current)*instruction;rgb_history=self.image(history)*instruction
        logits=base.static.belief_logits.float();relative=(logits-logits.amax(-1,keepdim=True)).clamp(-20,0)/5
        probability=logits.softmax(-1)*(logits.shape[-1]/784.)
        xy=static_inputs['grid_xy'].float();bounds=public['bounds'].float()
        pose=(public['pose_world'][:,:2]-bounds[:,:2])/(bounds[:,2:]-bounds[:,:2])
        spatial=(xy-pose[:,None]).norm(dim=-1) if self.use_pose else F.cosine_similarity(current,language[:,None],dim=-1)
        features=torch.stack((relative,probability,base.current_scores.float()/20,base.appearance_scores.float()/20,
            valid.any(-1).float(),remembered.any(-1).float(),spatial,known.float()),-1)
        quality=torch.cat((public['roi_quality'].float().flatten(-2),public['history_quality'].float().flatten(-2)),-1)
        x=torch.cat((self.geometry(base.static.dense_context.float()),rgb_current,rgb_history,quality,features),-1)
        raw,confidence=self.network(x).float().unbind(-1)
        centered=raw-(raw*known).sum(-1,keepdim=True)/known.sum(-1,keepdim=True).clamp_min(1) if self.centered else raw
        delta=self.bound*centered.tanh()*confidence.sigmoid()*known
        return delta,confidence.sigmoid(),known


def corrected_field(static,delta,grid_xy):
    if not delta.detach().any():return static
    logits=static.belief_logits.float()+delta
    ids=preserved_proposal_order(logits,static.candidate_ids)
    context=static.dense_context
    return GoalField(logits,ids,grid_xy.gather(1,ids[...,None].expand(-1,-1,2)),
        context.gather(1,ids[...,None].expand(-1,-1,context.shape[-1])),dense_context=context,
        visual_coverage=(delta!=0).float())


def correction_loss(logits,delta,grid_world,target_world,known,rank_weight=0.):
    distance=(grid_world-target_world[:,None,:2]).norm(dim=-1)
    legal=distance<=20.
    # If a coarse grid has no cell in 20m, retain the nearest cell for a finite loss.
    nearest=distance.argmin(-1)
    legal=legal.clone();legal[torch.arange(len(legal),device=legal.device),nearest]|=~legal.any(-1)
    updated=logits.float()+delta
    mass=torch.logsumexp(updated,-1)-torch.logsumexp(updated.masked_fill(~legal,-1e4),-1)
    old=torch.logsumexp(logits.float(),-1)-torch.logsumexp(logits.float().masked_fill(~legal,-1e4),-1)
    drift=(delta.square()*known).sum(-1)/known.sum(-1).clamp_min(1)
    good=updated.masked_fill(~legal,-1e4).amax(-1)
    bad=updated.masked_fill(legal,-1e4).amax(-1)
    ranking=F.softplus((bad-good)/.25).mean()*.25
    total=mass.mean()+.1*F.relu(mass-old).mean()+.02*drift.mean()+rank_weight*ranking
    return dict(total=total,mass=mass.mean(),baseline_mass=old.mean(),drift=drift.mean(),ranking=ranking)
