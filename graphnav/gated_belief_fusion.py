"""Experimental AerialVG candidate-attention -> reliability-gated *global* belief.

This does not modify the frozen Native Query Ranker, A0 prior, or production
navigation entry point. Public tensors only enter forward(); no target labels,
episode IDs, future frames, or map-specific ground truth are consulted.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

FEATURE_NAMES = (
    'null_probability', 'supported_visual_mass', 'a0_peak_probability',
    'a0_entropy', 'a0_top2_gap', 'visual_peak_probability',
    'visual_entropy', 'distribution_agreement', 'a0_at_visual_peak',
    'visual_at_a0_peak', 'peak_world_distance', 'query_valid_quality',
    'query_score_margin', 'supported_query_fraction', 'observed_grid_fraction',
    'visual_support_mass',
)


def _entropy(p):
    return -(p * p.clamp_min(1e-30).log()).sum(-1) / math.log(max(p.shape[-1], 2))


def visual_hypothesis(base_logits, query_logits, null_logits, kernel,
                      observed, valid, geometry, grid_world):
    """Construct a normalized candidate-attention visual goal distribution.

    Kernels have shape [B, grid, queries]; queries, valid [B, queries].
    Each supported query is normalized *over currently observed grid cells*.
    No evidence outside observed support is manufactured.
    """
    b, g = base_logits.shape
    if query_logits.ndim != 2:
        raise ValueError('Expected batched query scores')
    n = query_logits.shape[1]
    if (null_logits.shape != (b,) or query_logits.shape[0] != b
            or kernel.shape != (b, g, n) or observed.shape != (b, g)
            or valid.shape != (b, n) or geometry.shape != (b, n, 8)
            or grid_world.shape != (b, g, 2) or n == 0 or g < 2):
        raise ValueError('Invalid native-query fusion tensor shapes')
    if not (torch.isfinite(base_logits).all() and torch.isfinite(null_logits).all()
            and torch.isfinite(grid_world).all()
            and torch.isfinite(query_logits.masked_fill(~valid.bool(), 0)).all()
            and torch.isfinite(kernel).all() and torch.isfinite(geometry).all()):
        raise ValueError('Nonfinite public fusion input')
    if (kernel < 0).any():
        raise ValueError('Negative spatial evidence')
    observed = observed.bool()
    valid = valid.bool()
    masked_kernel = kernel.float() * observed[:, :, None] * valid[:, None, :]
    kernel_mass = masked_kernel.sum(1)
    supported = valid & (kernel_mass > 1e-9)
    normalized = masked_kernel / kernel_mass[:, None, :].clamp_min(1e-9)
    scores = query_logits.float().masked_fill(~valid, -1e4)
    posterior = torch.cat([scores, null_logits.float()[:, None]], -1).softmax(-1)
    supported_mass = (posterior[:, :-1] * supported).sum(-1)
    attention = (posterior[:, :-1] * supported) / supported_mass[:, None].clamp_min(1e-9)
    visual = torch.einsum('bgn,bn->bg', normalized, attention)
    usable = supported.any(-1) & observed.any(-1) & (supported_mass > 1e-9)
    visual = torch.where(usable[:, None], visual, torch.zeros_like(visual))
    visual = visual / visual.sum(-1, keepdim=True).clamp_min(1e-9)

    p0 = base_logits.float().softmax(-1)
    a0_idx = p0.argmax(-1)
    visual_idx = visual.argmax(-1)
    a0_top = p0.topk(2, dim=-1).values
    gap = a0_top[:, 0] - a0_top[:, 1]
    score_top = scores.masked_fill(~supported, -1e4).topk(min(2, n), dim=-1).values
    score_gap = (score_top[:, 0] - score_top[:, -1]).clamp(0, 30)
    score_gap = torch.where(supported.sum(-1) >= 2, score_gap, torch.zeros_like(score_gap))
    indices = torch.arange(b, device=p0.device)
    xy_distance = (grid_world[indices, a0_idx] - grid_world[indices, visual_idx]).norm(dim=-1)
    agreement = (p0 * visual).sum(-1) / (
        p0.square().sum(-1).sqrt() * visual.square().sum(-1).sqrt()
    ).clamp_min(1e-9)
    mean_quality = (
        geometry[:, :, 2].float().clamp(0, 1) * attention
    ).sum(-1)
    visual_peak = visual.max(-1).values
    visual_area = (visual > visual_peak[:, None] * 0.2) & observed
    features = torch.stack([
        posterior[:, -1], supported_mass,
        a0_top[:, 0], _entropy(p0), gap,
        visual_peak, _entropy(visual),
        agreement, p0[indices, visual_idx], visual[indices, a0_idx],
        (xy_distance / 100).clamp_max(4),
        mean_quality, score_gap / 10,
        supported.float().mean(-1), observed.float().mean(-1),
        (p0 * visual_area).sum(-1),
    ], -1)
    features = torch.where(usable[:, None], features, torch.zeros_like(features))
    if not torch.isfinite(features).all():
        raise ValueError('Nonfinite public reliability features')
    return dict(
        base_probability=p0, visual_probability=visual, attention=attention,
        features=features.detach(), usable=usable,
        supported=supported, null_probability=posterior[:, -1],
        supported_visual_mass=supported_mass,
    )


class ReliabilityGatedBelief(nn.Module):
    """Learn an accept/reject *reliability* score, not another object detector.

    Candidate attention is inherited from the frozen native query scores.
    A learned MLP gate chooses between global A0 and visual goal hypothesis.
    Unlike clipped per-grid residuals, a supported visual peak can displace
    the original A0 peak. Disabled/unavailable/rejected is bitwise exact A0.

    The training label is built outside this class with TRAIN target coordinates.
    """

    def __init__(self):
        super().__init__()
        self.reliability_net = nn.Sequential(
            nn.LayerNorm(len(FEATURE_NAMES)),
            nn.Linear(len(FEATURE_NAMES), 32), nn.GELU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.reliability_net[-1].weight)
        nn.init.constant_(self.reliability_net[-1].bias, -3.)

    def forward(self, base_logits, query_logits, null_logits, kernel,
                observed, valid, geometry, grid_world, *,
                enabled=False, threshold=0.8, transfer=0.85):
        if not 0 < threshold <= 1 or not 0 < transfer <= 1:
            raise ValueError('Gate threshold/transfer must belong to (0,1]')
        data = visual_hypothesis(
            base_logits, query_logits, null_logits, kernel,
            observed, valid, geometry, grid_world,
        )
        reliability_logit = self.reliability_net(data['features']).squeeze(-1)
        reliability = reliability_logit.sigmoid()
        accepted = data['usable'] & (reliability >= threshold) & (threshold < 1.)
        if not enabled:
            accepted = torch.zeros_like(accepted)
        strength = accepted.to(base_logits.dtype) * transfer
        mixture = ((1 - strength[:, None]) * data['base_probability']
                   + strength[:, None] * data['visual_probability'])
        # Retain original logit normalization and exact fallback for other
        # callers that inspect magnitude rather than only argmax/softmax.
        logits = mixture.clamp_min(1e-30).log() + torch.logsumexp(
            base_logits.float(), -1, keepdim=True
        )
        logits = torch.where(accepted[:, None], logits.to(base_logits.dtype), base_logits)
        return dict(
            logits=logits, accepted=accepted, gate=strength,
            reliability=reliability, reliability_logit=reliability_logit,
            **data,
        )


def training_gate_targets(base_logits, visual_probability, usable,
                          grid_world, target_world, radius_m=20., transfer=0.9):
    """TRAIN-only labels: accept iff the *planned global mixture* corrects A0.

    This does not claim instance matching. Unlike candidate-only ranking,
    the counterfactual uses the same transfer coefficient as evaluation.
    A0-correct states remain conservative negative gate targets.
    """
    if target_world.shape[0] != base_logits.shape[0] or target_world.shape[1] < 2:
        raise ValueError('Training target shape mismatch')
    if not 0 < transfer <= 1 or radius_m <= 0:
        raise ValueError('Invalid training counterfactual transfer')
    dist = (grid_world.float() - target_world[:, None, :2].float()).norm(dim=-1)
    before = dist.gather(1, base_logits.argmax(-1)[:, None]).squeeze(1) <= radius_m
    visual_correct = dist.gather(
        1, visual_probability.argmax(-1)[:, None]).squeeze(1) <= radius_m
    counterfactual = ((1 - transfer) * base_logits.float().softmax(-1)
                      + transfer * visual_probability)
    switched_correct = dist.gather(
        1, counterfactual.argmax(-1)[:, None]).squeeze(1) <= radius_m
    should_switch = (~before) & switched_correct & usable
    return dict(target=should_switch.float(), before=before,
                visual_correct=visual_correct,
                counterfactual_correct=switched_correct,
                oracle_eligible=should_switch)


def count_outcomes(base_logits, result_logits, grid_world, target_world, radius_m=20.):
    dist = (grid_world.float() - target_world[:, None, :2].float()).norm(dim=-1)
    a0 = dist.gather(1, base_logits.argmax(-1)[:, None]).squeeze(1)
    new = dist.gather(1, result_logits.argmax(-1)[:, None]).squeeze(1)
    return dict(
        count=len(a0), base_correct=int((a0 <= radius_m).sum()),
        updated_correct=int((new <= radius_m).sum()),
        corrected=int(((a0 > radius_m) & (new <= radius_m)).sum()),
        damaged=int(((a0 <= radius_m) & (new > radius_m)).sum()),
        base_ne_sum=float(a0.sum()), updated_ne_sum=float(new.sum()),
    )
