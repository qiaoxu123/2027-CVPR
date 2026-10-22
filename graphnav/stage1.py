"""Conservative V1 integration of existing A3, appearance and pretrained control.

All private labels remain in stage1_losses. Memory receives past public evidence.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from .appearance_matching import MultiScaleAppearanceMatcher
from .full_localization import GoalField
from .trajectory_policy import CandidateTrajectoryStop,trajectory_losses
from .static_localization import cross_entropy_goal


class QualityHistory(nn.Module):
    """Candidate-aligned confidence fusion of current and quality-weighted past views."""
    def __init__(self):
        super().__init__()
        self.quality=nn.Linear(6,1)
        nn.init.zeros_(self.quality.weight);nn.init.zeros_(self.quality.bias)

    def forward(self,current,current_valid,current_quality,past,past_valid,past_quality):
        cw=(current_quality*current_valid[...,None]).sum(-2)/current_valid.sum(-1).clamp_min(1)[...,None]
        hw=(past_quality*past_valid[...,None]).sum(-2)/past_valid.sum(-1).clamp_min(1)[...,None]
        gate=torch.sigmoid(self.quality(torch.cat((cw,hw),-1)).squeeze(-1))
        known=current_valid.any(-1);remembered=past_valid.any(-1)
        # Missing current views preserve historical support rather than erasing it.
        weight=torch.where(known,gate,torch.zeros_like(gate))
        weight=torch.where(remembered,weight,torch.ones_like(weight))
        scores=current*weight+past*(1-weight)
        return scores,known|remembered,weight


@dataclass
class Stage1Output:
    static: GoalField
    belief: GoalField
    trajectory: object
    appearance_scores: torch.Tensor
    observed: torch.Tensor
    current_scores: torch.Tensor
    current_observed: torch.Tensor
    history_weight: torch.Tensor | None
    delta: torch.Tensor


class GraphNavStage1(nn.Module):
    def __init__(self,a3,vision_dim=1024,matcher_hidden=64,residual_bound=.5):
        super().__init__();self.a3=a3
        self.matcher=MultiScaleAppearanceMatcher(vision_dim,matcher_hidden)
        self.history=QualityHistory();self.visual_gain=nn.Parameter(torch.zeros(()))
        self.trajectory=CandidateTrajectoryStop(hidden=a3.hidden)
        self.residual_bound=float(residual_bound);self.stage='A'

    def configure_stage(self,stage):
        if stage not in ('A','B','C'):raise ValueError('unknown stage')
        self.stage=stage;self.requires_grad_(True)
        self.a3.requires_grad_(stage=='C');self.trajectory.requires_grad_(stage=='C')
        # These original A3 branches are inactive in geometry mode. Retain
        # their weights without advertising them as optimized Stage1 capacity.
        for module in (self.a3.gnn.layer1,self.a3.gnn.layer2,self.a3.vision_proj,self.a3.visual_score):
            module.requires_grad_(False)
        self.history.requires_grad_(stage in ('B','C'))
        self.train(self.training)

    def train(self,mode=True):
        super().train(mode)
        if not any(p.requires_grad for p in self.a3.parameters()):self.a3.eval()
        if not any(p.requires_grad for p in self.trajectory.parameters()):self.trajectory.eval()
        return self

    def forward(self,*,static_inputs,tokens,token_mask,node_emb,native_text,appearance_text,
                roi_features,roi_valid,roi_quality,pose_world,bounds,ground_level,
                history_features=None,history_valid=None,history_quality=None,
                use_history=False,visual_control='real',base_field=None):
        if visual_control not in ('real','zero','shuffle','no-belief'):raise ValueError('unknown RGB intervention')
        if base_field is not None and any(p.requires_grad for p in self.a3.parameters()):raise ValueError('cannot cache trainable A3')
        static=base_field if base_field is not None else self.a3(static_inputs,variant='geometry',tokens=tokens,token_mask=token_mask,node_emb=node_emb)
        text=F.normalize(native_text.float()+appearance_text.float(),dim=-1)
        images=roi_features
        if visual_control=='zero':images=torch.zeros_like(images)
        if visual_control=='shuffle':
            images=images.clone()
            for i in range(len(images)):
                for scale in range(images.shape[2]):
                    ids=roi_valid[i,:,scale].nonzero().flatten()
                    if len(ids)>1:
                        permutation=torch.randperm(len(ids),generator=torch.Generator().manual_seed(0)).to(ids.device)
                        images[i,ids,scale]=images[i,ids[permutation],scale]
        current,known,_=self.matcher(text,images,roi_valid,roi_quality)
        scores=current;observed=known;gate=None
        if use_history:
            if any(x is None for x in (history_features,history_valid,history_quality)):raise ValueError('past-only history evidence required')
            history_images=history_features
            if visual_control=='zero':history_images=torch.zeros_like(history_images)
            if visual_control=='shuffle':
                history_images=history_images.clone()
                for i in range(len(history_images)):
                    for scale in range(history_images.shape[2]):
                        ids=history_valid[i,:,scale].nonzero().flatten()
                        if len(ids)>1:
                            permutation=torch.randperm(len(ids),generator=torch.Generator().manual_seed(0)).to(ids.device)
                            history_images[i,ids,scale]=history_images[i,ids[permutation],scale]
            past,_,_=self.matcher(text,history_images,history_valid,history_quality)
            scores,observed,gate=self.history(current,roi_valid,roi_quality,past,history_valid,history_quality)
        centered=scores-(scores*observed).sum(-1,keepdim=True)/observed.sum(-1,keepdim=True).clamp_min(1)
        delta=self.residual_bound*torch.tanh(self.visual_gain)*torch.tanh(centered/5.)*observed
        if visual_control=='no-belief':delta=torch.zeros_like(delta)
        logits=static.belief_logits.float()+delta
        if not observed.any() or visual_control=='no-belief' or (not self.training and self.visual_gain.detach()==0):
            # Preserve the original BF16 proposal ordering and context at fallback.
            field=static
        else:
            ids=preserved_proposal_order(logits,static.candidate_ids)
            ids=torch.where(observed.any(-1,keepdim=True),ids,static.candidate_ids)
            dense=static.dense_context
            field=GoalField(logits,ids,static_inputs['grid_xy'].gather(1,ids[...,None].expand(-1,-1,2)),
                dense.gather(1,ids[...,None].expand(-1,-1,dense.shape[-1])),dense_context=dense,visual_coverage=observed.float())
        probability=field.belief_logits.float().softmax(-1)
        trajectory=self.trajectory(candidate_xy=field.candidate_xy.float(),candidate_context=field.candidate_context.half().float(),
            candidate_logits=field.belief_logits.gather(1,field.candidate_ids).float(),pose_world=pose_world,bounds=bounds,
            ground_level=ground_level,belief_entropy=-(probability*probability.clamp_min(1e-8).log()).sum(-1)/math.log(logits.shape[-1]),
            observed_fraction=torch.zeros(len(logits),device=logits.device))
        return Stage1Output(static,field,trajectory,scores,observed,current,known,gate,delta)


def stage1_losses(output,*,grid_world,target_world,pose_world,teacher_world,valid_teacher_z,bounds,
                  observed_scales,teacher_logits=None,stage='A',goal_sigma_cells=.9):
    distance=(grid_world-target_world[:,None,:2]).norm(dim=-1)
    target=(target_world[:,:2]-bounds[:,:2])/(bounds[:,2:]-bounds[:,:2]);grid_size=int(math.sqrt(grid_world.shape[1]))
    goal=cross_entropy_goal(output.static.belief_logits,target,grid_size=grid_size,sigma_cells=goal_sigma_cells)
    belief=cross_entropy_goal(output.belief.belief_logits,target,grid_size=grid_size,sigma_cells=goal_sigma_cells)
    positive,negative=visual_supervision_regions(grid_world,target_world,observed_scales)
    positive=positive&output.observed;negative=negative&output.observed
    eligible=positive.any(-1)&negative.any(-1)
    scores=output.appearance_scores.float()
    visual_terms=torch.logsumexp(scores.masked_fill(~(positive|negative),-1e4),-1)-torch.logsumexp(scores.masked_fill(~positive,-1e4),-1)
    visual=visual_terms[eligible].mean() if eligible.any()else scores.sum()*0
    # Hard visible alternatives selected by appearance, with every legal target-radius cell positive.
    hard_negative=scores.masked_fill(~negative,-1e4).amax(-1)
    legal=scores.masked_fill(~positive,-1e4).amax(-1)
    rank=F.relu(.2+hard_negative-legal)[eligible].mean() if eligible.any()else scores.sum()*0
    ids=output.belief.candidate_ids;candidate_world=grid_world.gather(1,ids[...,None].expand(-1,-1,2))
    for key in ('paths_world','mode_logits','candidate_logits','stop_logits'):setattr(output.trajectory,key,getattr(output.trajectory,key).float())
    nav=trajectory_losses(output.trajectory,candidate_world_xy=candidate_world,pose_world=pose_world,target_world=target_world,
                          teacher_world=teacher_world,valid_teacher_z=valid_teacher_z,stop_pos_weight=torch.tensor(20.,device=scores.device))
    anchor=F.kl_div(output.static.belief_logits.float().log_softmax(-1),teacher_logits.float().softmax(-1),reduction='batchmean') if teacher_logits is not None else goal*0
    total=goal+.1*visual+belief+.05*rank
    if stage=='C':total=total+.1*nav['path']+.02*nav['mode']+.05*nav['ranking']+.05*nav['stop']+anchor
    return dict(total=total,goal=goal,visual=visual,belief=belief,rank=rank,trajectory=nav['path'],trajectory_mode=nav['mode'],
                trajectory_rank=nav['ranking'],stop=nav['stop'],anchor=anchor,visual_eligible=eligible.float().mean())


def visual_supervision_regions(grid_world,target_world,observed_scales):
    """Loss-only labels: exclude an alternative if ANY used ROI may contain GT.

    Native contextual pooling uses 32/64/128m ROIs. Being outside the 20m
    navigation radius does not imply its crop excludes the same target object.
    Ambiguous overlap is ignored; invisible cells are never negatives.
    """
    extent=grid_world.new_tensor([32.,64.,128.])/2
    delta=(grid_world-target_world[:,None,:2]).abs()
    contains=((delta[:,:,None,:]<=extent[None,None,:,None]).all(-1)&observed_scales).any(-1)
    legal=delta.norm(dim=-1)<=20.
    return contains&legal,observed_scales.any(-1)&~contains&~legal


def preserved_proposal_order(logits,base_ids):
    """Update scores while preserving original controller proposal ties.

    BF16 topk and FP32 topk can break equal-score ties differently. A zero
    visual residual must not change controller candidates merely by casting.
    """
    full=torch.arange(logits.shape[-1],device=logits.device)[None].expand(len(logits),-1)
    remaining=full[~(full[:,:,None]==base_ids[:,None,:]).any(-1)].reshape(len(logits),-1)
    priority=torch.cat((base_ids,remaining),-1)
    order=logits.gather(1,priority).argsort(dim=-1,descending=True,stable=True)[:,:base_ids.shape[-1]]
    return priority.gather(1,order)
