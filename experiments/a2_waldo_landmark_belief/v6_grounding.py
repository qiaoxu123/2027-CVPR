"""A2-V6 experimental relational candidate reranker and *disabled-by-default* correction.

No private target coordinate, association label or navigation result enters forward().
All geometry and RGB features must come from the verified, observed public cache.
This is an isolated TRAIN/DEV pilot; not an admitted CityNav runtime.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .relation_v3_fusion import candidate_kernels
from .revision_fusion import batch_belief_mix

MAX_ALPHA = 0.10


def _check_public(network, spatial, evidence):
    required_network = ("box_rgb", "box_valid", "box_kind", "pair_geometry", "pair_valid")
    required_spatial = ("base_logits", "grid_world", "candidate_xy", "candidate_support")
    if any(key not in network for key in required_network):
        raise ValueError("Missing V6 public network tensors")
    if any(key not in spatial for key in required_spatial):
        raise ValueError("Missing V6 public spatial tensors")
    rgb = network["box_rgb"]
    b, n, d = rgb.shape
    if d != 1024 or network["pair_geometry"].shape[:2] != (b, n):
        raise ValueError("V6 public feature dimensions changed")
    if network["pair_geometry"].shape[-1] != 14:
        raise ValueError("Expected fourteen public reference geometry channels")
    if spatial["candidate_support"].shape != (b, n, spatial["base_logits"].shape[-1]):
        raise ValueError("V6 candidate support and A0 grid differ")
    if spatial["candidate_xy"].shape != (b, n, 2):
        raise ValueError("V6 candidate XY axes changed")
    if evidence["scores"].shape != (b, n) or evidence["reliable"].shape != (b, n):
        raise ValueError("Frozen evidence axes changed")
    tensors = (rgb, network["pair_geometry"], spatial["base_logits"], spatial["candidate_xy"])
    if not all(torch.isfinite(x).all() for x in tensors):
        raise ValueError("Nonfinite public V6 inputs")


def public_candidate_features(network, spatial, evidence):
    """Per-candidate, label-free A0 context and verified landmark geometry."""
    _check_public(network, spatial, evidence)
    support = spatial["candidate_support"].bool()
    reliable = evidence["reliable"].bool()
    valid = network["box_valid"].bool() & reliable & support.any(-1)
    pair_valid = network["pair_valid"].bool()
    geom = network["pair_geometry"].float()
    pair_mask = pair_valid[..., None]
    count = pair_valid.sum(2, keepdim=True).clamp_min(1)
    mean_geometry = (geom * pair_mask).sum(2) / count
    max_geometry = geom.masked_fill(~pair_mask, -1e4).max(2).values
    max_geometry = torch.where(pair_valid.any(2)[..., None], max_geometry, torch.zeros_like(max_geometry))

    base = spatial["base_logits"].float()
    p0 = base.softmax(-1)
    prior_mass = (p0[:, None] * support).sum(-1)
    prior_peak = p0[:, None].masked_fill(~support, 0).max(-1).values
    peak_xy = spatial["grid_world"].float().gather(
        1, base.argmax(-1)[:, None, None].expand(-1, 1, 2)
    ).squeeze(1)
    distance = (spatial["candidate_xy"].float() - peak_xy[:, None]).norm(dim=-1)
    entropy = -(p0 * p0.clamp_min(1e-30).log()).sum(-1) / math.log(max(p0.shape[-1], 2))
    score = evidence["scores"].float().masked_fill(~valid, 0).clamp(-30, 30)
    trust = evidence["candidate_match_logits"].float().masked_fill(~valid, 0).clamp(-30, 30)
    scalar = torch.stack(
        (prior_mass, prior_peak, distance.clamp_max(400) / 100,
         support.float().mean(-1), entropy[:, None].expand_as(distance),
         score / 10, trust / 10, pair_valid.any(2).float(),
         network["box_kind"].float()), dim=-1
    )
    return mean_geometry, max_geometry, scalar, valid


class RelationalGrounder(nn.Module):
    """Frozen-evidence residual ranking with an 8-neighbor spatial relation graph.

    Does not claim verified object identity: TRAIN ranking labels measure spatial
    utility within 20 m; unknown candidate labels remain outside the loss.
    """

    def __init__(self, frozen_evidence, hidden=64, neighbors=8):
        super().__init__()
        if not 1 <= neighbors <= 64:
            raise ValueError("Invalid graph degree")
        self.evidence = frozen_evidence
        for p in self.evidence.parameters():
            p.requires_grad_(False)
        self.neighbors = neighbors
        self.visual = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 32), nn.GELU())
        self.embed = nn.Sequential(nn.Linear(32 + 28 + 9, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.value = nn.Linear(hidden, hidden)
        self.edges = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, 1))
        self.output = nn.Linear(hidden * 2, 1)
        self.coverage = nn.Sequential(nn.Linear(hidden * 2 + 2, 32),
                                      nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        nn.init.constant_(self.coverage[-1].bias, -2.0)

    def forward(self, network, spatial):
        # Training this wrapper never updates or turns on dropout in the source model.
        self.evidence.eval()
        with torch.no_grad():
            evidence = self.evidence(**network)
        mean_geo, max_geo, scalar, valid = public_candidate_features(network, spatial, evidence)
        b, n = valid.shape
        appearance = self.visual(network["box_rgb"].float())
        h = self.embed(torch.cat((appearance, mean_geo, max_geo, scalar), -1))
        xy = spatial["candidate_xy"].float()
        delta = xy[:, None, :, :] - xy[:, :, None, :]
        dist = delta.norm(dim=-1)
        same_kind = (network["box_kind"][:, :, None] ==
                     network["box_kind"][:, None, :]).float()
        edge_features = torch.cat((delta.clamp(-400, 400) / 40,
                                   dist.clamp_max(400)[..., None] / 40,
                                   same_kind[..., None]), -1)
        neighbor_dist = dist.masked_fill(~valid[:, None, :], float("inf"))
        nearest = neighbor_dist.topk(min(self.neighbors, n), dim=-1, largest=False).indices
        neighborhood = torch.zeros((b, n, n), dtype=torch.bool, device=xy.device)
        neighborhood.scatter_(2, nearest, True)
        edge_valid = neighborhood & valid[:, :, None] & valid[:, None, :]
        attention_logits = (self.query(h) @ self.key(h).transpose(1, 2)) / math.sqrt(h.shape[-1])
        attention_logits = attention_logits + self.edges(edge_features).squeeze(-1)
        attention = attention_logits.masked_fill(~edge_valid, -1e4).softmax(-1)
        attention = attention * edge_valid
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-8)
        combined = torch.cat((h, attention @ self.value(h)), -1)
        residual = 4 * torch.tanh(self.output(combined).squeeze(-1))
        raw = evidence["scores"].float().detach()
        logits = (raw + residual).masked_fill(~valid, -1e4)

        weights = valid.float() / valid.sum(-1, keepdim=True).clamp_min(1)
        pooled = (combined * weights[..., None]).sum(1)
        top2 = logits.masked_fill(~valid, -1e4).topk(min(n, 2), dim=-1).values
        margin = top2[:, 0] - top2[:, -1]
        margin = torch.where(valid.sum(-1) >= 2, margin, torch.zeros_like(margin))
        prior = spatial["base_logits"].float().softmax(-1)
        entropy = -(prior * prior.clamp_min(1e-30).log()).sum(-1)
        entropy = entropy / math.log(max(prior.shape[-1], 2))
        coverage_logit = self.coverage(torch.cat((pooled, margin[:, None] / 10,
                                                  entropy[:, None]), -1)).squeeze(-1)
        return {"logits": logits, "valid": valid, "coverage_logit": coverage_logit,
                "margin": margin, "evidence": evidence}


def candidate_proposal(spatial, prediction, temperature=0.5):
    """Normalize verified candidates over the *entire* inference candidate pool."""
    if not 0 < temperature <= 5 or not math.isfinite(temperature):
        raise ValueError("Invalid V6 proposal temperature")
    valid = prediction["valid"].bool()
    scores = prediction["logits"]
    trust = prediction["evidence"]["candidate_match_logits"].float().detach()
    available = valid.any(-1)
    safe_logits = (scores + F.logsigmoid(trust)) / temperature
    safe_logits = safe_logits.masked_fill(~valid, -1e4)
    weights = safe_logits.softmax(-1) * valid
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
    kernels = candidate_kernels(spatial)
    proposal = (weights[..., None] * kernels).sum(1)
    return proposal, available


def selective_belief_update(spatial, prediction, *, enabled=False,
                            alpha=MAX_ALPHA, minimum_coverage=0.95,
                            minimum_margin=2.0, minimum_entropy=0.45):
    """Exact A0 fallback when disabled, unsupported, or uncertain.

    Enabling is prohibited in the formal runtime until independently admitted.
    No GT, target ID or private labels are accepted by this function.
    """
    base = spatial["base_logits"]
    if not 0 <= alpha <= MAX_ALPHA:
        raise ValueError("V6 correction exceeds bounded alpha")
    if not 0 <= minimum_coverage <= 1 or minimum_margin < 0 or not 0 <= minimum_entropy <= 1:
        raise ValueError("Invalid immutable V6 gate thresholds")
    if not enabled or alpha == 0:
        return base, torch.zeros(base.shape[0], dtype=torch.bool, device=base.device)
    proposal, available = candidate_proposal(spatial, prediction)
    p0 = base.float().softmax(-1)
    entropy = -(p0 * p0.clamp_min(1e-30).log()).sum(-1)
    entropy = entropy / math.log(max(p0.shape[-1], 2))
    accepted = (available & (prediction["coverage_logit"].sigmoid() >= minimum_coverage)
                & (prediction["margin"] >= minimum_margin) & (entropy >= minimum_entropy))
    mixed = batch_belief_mix(base.float(), proposal, torch.full_like(entropy, alpha))
    return torch.where(accepted[:, None], mixed.to(base.dtype), base), accepted


def weak_spatial_ranking_loss(prediction, labels):
    """Loss on validated spatial-utility candidates, NEVER identity ground truth.

    All unlabeled/unsupported boxes are excluded from the denominator; known
    spatial negatives participate alongside known positives.
    """
    valid = prediction["valid"] & labels["utility_valid"].bool()
    positive = valid & labels["localization_utility"].bool()
    contrast = positive.any(-1) & (valid & ~positive).any(-1)
    scores = prediction["logits"].masked_fill(~valid, -1e4)
    normalized_target = positive.float() / positive.sum(-1, keepdim=True).clamp_min(1)
    logp = scores.log_softmax(-1).masked_fill(~positive, 0)
    rank_rows = -(normalized_target * logp).sum(-1)
    weights = labels["sample_weight"].float()
    if not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Invalid source-balanced TRAIN weights")
    normalized = weights / weights.mean().clamp_min(1e-8)
    ranking = (rank_rows * normalized * contrast).sum() / (normalized * contrast).sum().clamp_min(1)
    # This head predicts presence of a usable *spatial* candidate, not visible
    # target identity. It is not trained on sealed or formal benchmark roles.
    coverage_rows = F.binary_cross_entropy_with_logits(
        prediction["coverage_logit"], positive.any(-1).float(), reduction="none"
    )
    coverage = (coverage_rows * normalized).mean()
    return ranking + 0.25 * coverage, {
        "ranking": ranking.detach(), "coverage": coverage.detach(),
        "contrast_rows": contrast.sum().detach(),
        "positive_rows": positive.any(-1).sum().detach(),
    }


@torch.no_grad()
def oracle_diagnostic(spatial, labels, prediction):
    """Private TRAIN/DEV-only upper-bound diagnostic; NEVER a runtime action."""
    grid = spatial["grid_world"].float()
    target = labels["target_xy"].float()
    distance = (grid - target[:, None]).norm(dim=-1)
    kernels = candidate_kernels(spatial)
    positive = (labels["localization_utility"].bool() & labels["utility_valid"].bool()
                & prediction["valid"])
    kernel_peak = kernels.argmax(-1)
    proposal_ne = distance.gather(1, kernel_peak)
    oracle_dist = proposal_ne.masked_fill(~positive, float("inf"))
    best = oracle_dist.argmin(-1)
    covered = positive.any(-1)
    oracle_q = kernels.gather(1, best[:, None, None].expand(-1, 1, kernels.shape[-1])).squeeze(1)
    oracle_mix = batch_belief_mix(spatial["base_logits"].float(), oracle_q,
                                 torch.full_like(covered.float(), MAX_ALPHA))
    base_ne = distance.gather(1, spatial["base_logits"].argmax(-1)[:, None]).squeeze(1)
    oracle_ne = distance.gather(1, oracle_q.argmax(-1)[:, None]).squeeze(1)
    mixed_ne = distance.gather(1, oracle_mix.argmax(-1)[:, None]).squeeze(1)
    recoverable = covered & (base_ne > 20) & (oracle_ne <= 20)
    return {
        "views": int(covered.numel()),
        "covered": int(covered.sum()),
        "a0_wrong_and_oracle_usable": int(recoverable.sum()),
        "oracle_correct_on_covered": int(((oracle_ne <= 20) & covered).sum()),
        "fixed_alpha_oracle_corrections": int(((base_ne > 20) & (mixed_ne <= 20) & covered).sum()),
        "fixed_alpha_oracle_breaks": int(((base_ne <= 20) & (mixed_ne > 20) & covered).sum()),
    }
