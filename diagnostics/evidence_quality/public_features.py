"""Public-only, isolated evidence-quality diagnostic; never a live policy."""
import torch
from torch.nn import functional as F

from experiments.a2_waldo_landmark_belief.relation_v3_fusion import proposal_with_quality
from experiments.a2_waldo_landmark_belief.risk_probe import public_risk_features


EXTRA_NAMES = (
    'log_candidate_count', 'log_reference_count', 'candidate_weight_entropy',
    'candidate_weight_margin', 'weighted_detection_confidence',
    'weighted_projection_coverage', 'weighted_height_validity',
)


@torch.no_grad()
def public_features(*, base, grid, candidate_xy, candidate_support, scores,
                    match_logits, reliable, box_valid, reference_valid,
                    pair_valid, pair_geometry):
    """Compute the fixed 15 original + 7 added features, without private labels.

    Exact keyword arguments deliberately exclude targets, source identities,
    outcomes and previously computed private candidate-coverage masks.
    """
    if scores.ndim != 2 or reference_valid.ndim != 2 or base.ndim != 2:
        raise ValueError('Expected batched public evidence')
    b, n = scores.shape
    l, q = reference_valid.shape[1], base.shape[1]
    expected = (
        (base, (b, q)), (grid, (b, q, 2)), (candidate_xy, (b, n, 2)),
        (candidate_support, (b, n, q)), (match_logits, (b, n)),
        (reliable, (b, n)), (box_valid, (b, n)),
        (reference_valid, (b, l)), (pair_valid, (b, n, l)),
        (pair_geometry, (b, n, l, 14)),
    )
    if min(b, n, l, q) < 1 or any(tuple(t.shape) != shape for t, shape in expected):
        raise ValueError('Public evidence axes changed; empty axes need masked padding')
    tensors = (scores, *(t for t, _ in expected))
    if any(t.device != scores.device or not torch.isfinite(t).all() for t in tensors):
        raise ValueError('Nonfinite or mixed-device public evidence')
    masks = (candidate_support, reliable, box_valid, reference_valid, pair_valid)
    if any(t.dtype != torch.bool for t in masks):
        raise ValueError('Public validity masks must be boolean')
    active = pair_valid & box_valid[:, :, None] & reference_valid[:, None, :]
    if not torch.equal(reliable, active.any(-1)):
        raise ValueError('Reliability differs from original public relation mask')

    spatial = dict(candidate_xy=candidate_xy, grid_world=grid,
                   candidate_support=candidate_support)
    proposal, available, quality = proposal_with_quality(
        spatial, scores, match_logits, reliable)
    original = public_risk_features(base, proposal, grid, quality, available)
    eligible = reliable & candidate_support.any(-1)
    logits = (scores.float() + F.logsigmoid(match_logits.float())).masked_fill(
        ~eligible, -torch.inf)
    safe = torch.where(available[:, None], logits, torch.zeros_like(logits))
    weights = safe.softmax(-1) * eligible
    count = eligible.sum(-1).float()
    entropy = -(weights * weights.clamp_min(1e-30).log()).sum(-1)
    entropy = entropy / count.clamp_min(2).log()
    top = weights.topk(min(n, 2), dim=-1).values
    margin = top[:, 0] - (top[:, 1] if n > 1 else torch.zeros_like(top[:, 0]))

    def weighted_channel(index):
        # Select valid relationships before averaging; padding is not evidence.
        values = pair_geometry[..., index].float().masked_fill(~active, 0)
        per_candidate = values.sum(-1) / active.sum(-1).clamp_min(1)
        return (per_candidate * weights).sum(-1)

    extras = torch.stack((count.log1p(), reference_valid.sum(-1).float().log1p(),
                          entropy, margin, weighted_channel(9),
                          weighted_channel(13), weighted_channel(11)), -1)
    extras = extras * available[:, None]
    result = torch.cat((original, extras), -1)
    if not torch.isfinite(result).all():
        raise ValueError('Nonfinite diagnostic feature output')
    return result
