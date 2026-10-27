"""Joint A3/observed appearance/belief/trajectory/Stop, with label-free forward.

Frozen backbone features may be cached. A3 outputs MUST be recomputed while
training. The independent original A3 implementation and checkpoints are intact.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .appearance_matching import MultiScaleAppearanceMatcher
from .full_localization import GoalField
from .trajectory_policy import CandidateTrajectoryStop, trajectory_losses


@dataclass
class JointOutput:
    static: GoalField
    belief: GoalField
    trajectory: object
    appearance_scores: torch.Tensor
    observed: torch.Tensor


class JointActiveGraphNav(nn.Module):
    def __init__(self, a3, *, vision_dim=1024, matcher_hidden=64):
        super().__init__()
        self.a3 = a3  # trainable; never detach its output
        self.matcher = MultiScaleAppearanceMatcher(vision_dim, matcher_hidden)
        self.visual_gain = nn.Parameter(torch.zeros(()))
        self.trajectory = CandidateTrajectoryStop(hidden=a3.hidden)

    def forward(self, *, static_inputs, tokens, token_mask, node_emb,
                native_text, roi_features, roi_valid, roi_quality, pose_world,
                bounds, ground_level, history_scores=None, history_observed=None,
                visual_control='full'):
        static = self.a3(static_inputs, variant='geometry', tokens=tokens,
                         token_mask=token_mask, node_emb=node_emb)
        scores, known, _ = self.matcher(native_text, roi_features, roi_valid, roi_quality)
        if scores.shape != static.belief_logits.shape:
            raise ValueError('appearance must align with the full global candidate grid')
        if (history_scores is None) != (history_observed is None):
            raise ValueError('history scores and observed mask must be supplied together')
        if history_scores is not None:
            if history_scores.shape != scores.shape or history_observed.shape != known.shape:
                raise ValueError('history candidate alignment mismatch')
            # Both absent remains unknown; a missing current observation does not erase history.
            counts = known.float() + history_observed.float()
            scores = (scores * known + history_scores * history_observed) / counts.clamp_min(1)
            known = known | history_observed
        # Observed-only positive support: no penalty for absence of observation.
        # Centering only on supported cells prevents the native shared-space bias
        # from promoting every visible cell equally. Initial gain=0 preserves A3.
        centered = scores - (scores * known).sum(-1, keepdim=True) / known.sum(-1, keepdim=True).clamp_min(1)
        if visual_control not in ('full', 'no-belief'):
            raise ValueError('unknown visual-belief ablation')
        # No-belief keeps the same observation coverage and trajectory inputs,
        # isolating the causal effect of RGB appearance on goal belief.
        delta = (torch.tanh(self.visual_gain) * centered * known
                 if visual_control == 'full' else torch.zeros_like(centered))
        logits = static.belief_logits.float() + delta
        ids = logits.topk(min(self.a3.top_k, logits.shape[-1]), -1).indices
        grid = static_inputs['grid_xy']
        dense = static.dense_context
        field = GoalField(logits, ids, grid.gather(1, ids[..., None].expand(-1,-1,2)),
                          dense.gather(1,ids[..., None].expand(-1,-1,dense.shape[-1])),
                          visual_coverage=known.float(), dense_context=dense)
        probability = logits.softmax(-1)
        trajectory = self.trajectory(candidate_xy=field.candidate_xy,
            candidate_context=field.candidate_context,
            candidate_logits=logits.gather(1,ids), pose_world=pose_world,
            bounds=bounds, ground_level=ground_level,
            belief_entropy=-(probability*probability.clamp_min(1e-9).log()).sum(-1)/math.log(logits.shape[-1]),
            observed_fraction=known.float().mean(-1))
        return JointOutput(static,field,trajectory,scores,known)


def joint_losses(output, *, grid_world, target_world, pose_world, teacher_world,
                 valid_teacher_z):
    """All destination/future quantities are supervision only, outside forward."""
    dist = (grid_world - target_world[:,None,:2]).norm(dim=-1)
    label = (-dist.square()/(2*20.**2)).softmax(-1)
    static = -(label * output.static.belief_logits.float().log_softmax(-1)).sum(-1).mean()
    belief = -(label * output.belief.belief_logits.float().log_softmax(-1)).sum(-1).mean()
    # Supervise actual observed alternatives; no invisible target as a negative.
    available = output.observed
    visual_label = label * available
    usable = visual_label.sum(-1) > .1
    visual_label = visual_label / visual_label.sum(-1,keepdim=True).clamp_min(1e-9)
    visual_ce = -(visual_label * output.appearance_scores.masked_fill(~available,-1e4).log_softmax(-1)).sum(-1)
    visual = visual_ce[usable].mean() if usable.any() else output.appearance_scores.sum()*0
    ids = output.belief.candidate_ids
    candidate_world = grid_world.gather(1, ids[...,None].expand(-1,-1,2))
    trajectory = output.trajectory
    # Keep metric-coordinate loss reductions FP32 under BF16 training.
    for key in ('paths_world','mode_logits','candidate_logits','stop_logits'):
        setattr(trajectory,key,getattr(trajectory,key).float())
    navigation = trajectory_losses(trajectory,candidate_world_xy=candidate_world,
        pose_world=pose_world,target_world=target_world,teacher_world=teacher_world,
        valid_teacher_z=valid_teacher_z,stop_pos_weight=torch.tensor(20.,device=dist.device))
    return dict(total=.5*static+.5*belief+.1*visual+navigation['total'],
                static=static,belief=belief,visual=visual,
                **{'trajectory_'+key:value for key,value in navigation.items() if key!='total'})
