"""Opt-in joint RGB+language+geometry goal localizer; no visual logit residual.

The same frozen SigLIP2 ROI observations and causal history as GraphNav-V1
are incorporated into each candidate's pre-attention query. This model is
experimental and must be trained with a dedicated checkpoint/protocol.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

from .stage1 import GraphNavStage1, Stage1Output, preserved_proposal_order
from .full_localization import GoalField
from .trajectory_policy import CandidateTrajectoryStop


def _pooled_roi(images, valid, quality):
    """Quality-weighted frozen SigLIP vectors; missing views contribute zero."""
    weights = quality.float()[..., 0].clamp(0, 1) * quality.float()[..., 1].clamp(0, 1)
    weights = weights * valid.float()
    vectors = (images.float() * weights[..., None]).sum(-2)
    vectors = vectors / weights.sum(-1, keepdim=True).clamp_min(1e-8)
    return vectors * valid.any(-1)[..., None]


def _shuffle_observed(images, valid):
    """Matched deterministic intervention, preserving position and valid masks."""
    images = images.clone()
    generator = torch.Generator().manual_seed(0)
    for i in range(len(images)):
        for scale in range(images.shape[2]):
            ids = valid[i, :, scale].nonzero().flatten()
            if len(ids) > 1:
                perm = torch.randperm(len(ids), generator=generator).to(ids.device)
                images[i, ids, scale] = images[i, ids[perm], scale]
    return images


class JointRGBGeometryStage1(GraphNavStage1):
    """All RGB influence is upstream of token/landmark attention and Top-K.

    Original A3, trajectory/Stop, matcher and history interfaces are preserved.
    No independent post-hoc appearance-to-logit correction is applied.
    """

    def __init__(self, a3, vision_dim=1024, matcher_hidden=64, residual_bound=.5):
        super().__init__(a3, vision_dim, matcher_hidden, residual_bound)
        if hasattr(self.a3, "rgb_query"):
            raise ValueError("unexpected existing RGB fusion adapter")
        self.a3.rgb_query = nn.Sequential(
            nn.LayerNorm(vision_dim + 1),
            nn.Linear(vision_dim + 1, a3.hidden, bias=False),
        )
        nn.init.normal_(self.a3.rgb_query[1].weight, mean=0., std=0.001)

    def configure_stage(self, stage):
        super().configure_stage(stage)
        # The legacy residual gain is intentionally dead in this architecture.
        self.visual_gain.requires_grad_(False)

    def forward(self, *, static_inputs, tokens, token_mask, node_emb,
                native_text, appearance_text, roi_features, roi_valid, roi_quality,
                pose_world, bounds, ground_level, history_features=None,
                history_valid=None, history_quality=None, use_history=False,
                visual_control='real', base_field=None):
        if visual_control not in ('real', 'zero', 'shuffle', 'no-belief'):
            raise ValueError('unknown RGB intervention')
        if base_field is not None and any(p.requires_grad for p in self.a3.parameters()):
            raise ValueError('cannot cache a trainable localizer')

        # Exact same public RGB data and visual-supervision matcher as formal V1.
        now = roi_features
        past = history_features
        if visual_control == 'zero':
            now = torch.zeros_like(now)
            if past is not None:
                past = torch.zeros_like(past)
        elif visual_control == 'shuffle':
            now = _shuffle_observed(now, roi_valid)
            if past is not None and history_valid is not None:
                past = _shuffle_observed(past, history_valid)

        text = F.normalize(native_text.float() + appearance_text.float(), dim=-1)
        current, current_known, _ = self.matcher(text, now, roi_valid, roi_quality)
        now_vector = _pooled_roi(now, roi_valid, roi_quality)
        observed = current_known
        scores = current
        gate = None
        fused_vector = now_vector
        if use_history:
            if any(v is None for v in (past, history_valid, history_quality)):
                raise ValueError('past-only history evidence required')
            history_scores, _, _ = self.matcher(text, past, history_valid, history_quality)
            scores, observed, gate = self.history(
                current, roi_valid, roi_quality,
                history_scores, history_valid, history_quality,
            )
            old_vector = _pooled_roi(past, history_valid, history_quality)
            fused_vector = gate[..., None] * now_vector + (1 - gate[..., None]) * old_vector

        fused_vector = fused_vector * observed[..., None]
        # The auxiliary matcher is inside the joint representation; it never
        # updates candidate logits independently.
        rgb_query = torch.cat((fused_vector, scores[..., None] / 100.), -1)

        static = base_field if base_field is not None else self.a3(
            static_inputs, variant='geometry',
            tokens=tokens, token_mask=token_mask, node_emb=node_emb,
        )
        if not observed.any() or visual_control == 'no-belief':
            field = static
        else:
            joint = self.a3(
                static_inputs, variant='geometry',
                tokens=tokens, token_mask=token_mask, node_emb=node_emb,
                candidate_rgb=rgb_query, candidate_rgb_valid=observed,
            )
            # For unseen cells the *direct* logit/context is unchanged. No
            # image-null-as-negative update is permitted.
            logits = torch.where(observed, joint.belief_logits, static.belief_logits)
            dense = torch.where(
                observed[..., None], joint.dense_context, static.dense_context,
            )
            ids = preserved_proposal_order(logits, static.candidate_ids)
            ids = torch.where(
                observed.any(-1, keepdim=True), ids, static.candidate_ids,
            )
            field = GoalField(
                belief_logits=logits,
                candidate_ids=ids,
                candidate_xy=static_inputs['grid_xy'].gather(
                    1, ids[..., None].expand(-1, -1, 2)),
                candidate_context=dense.gather(
                    1, ids[..., None].expand(-1, -1, dense.shape[-1])),
                dense_context=dense,
                visual_coverage=observed.float(),
            )

        probability = field.belief_logits.float().softmax(-1)
        trajectory = self.trajectory(
            candidate_xy=field.candidate_xy.float(),
            candidate_context=field.candidate_context.half().float(),
            candidate_logits=field.belief_logits.gather(1, field.candidate_ids).float(),
            pose_world=pose_world, bounds=bounds, ground_level=ground_level,
            belief_entropy=-(probability * probability.clamp_min(1e-8).log()).sum(-1)
                / math.log(field.belief_logits.shape[-1]),
            observed_fraction=torch.zeros(len(probability), device=probability.device),
        )
        delta = field.belief_logits.float() - static.belief_logits.float()
        return Stage1Output(
            static, field, trajectory, scores, observed,
            current, current_known, gate, delta,
        )
