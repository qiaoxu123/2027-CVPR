"""Experimental incremental gate: rejected proposals preserve frozen Native."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .gated_belief_fusion import visual_hypothesis


class NativePreservingGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(20), nn.Linear(20, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, native_logits, relation_query, null, kernel, observed,
                valid, geometry, grid_world, has_relation, *, enabled=False,
                threshold=1., slope=1., bias=0., transfer=.9):
        if not 0 < threshold <= 1 or not 0 < transfer <= 1 or slope <= 0:
            raise ValueError('Invalid gate/calibration parameters')
        data = visual_hypothesis(native_logits, relation_query, null, kernel,
                                 observed, valid, geometry, grid_world)
        p = data['base_probability']
        proposal = (1-transfer)*p + transfer*data['visual_probability']
        proposal_logits = proposal.clamp_min(1e-30).log() + torch.logsumexp(native_logits.float(), -1, keepdim=True)
        indices = torch.arange(len(p), device=p.device)
        ni, pi = p.argmax(-1), proposal.argmax(-1)
        extra = torch.stack([p[indices, pi], proposal[indices, ni],
                             (p-proposal).abs().sum(-1), has_relation.float()], -1)
        raw = self.net(torch.cat([data['features'], extra], -1).detach()).squeeze(-1)
        probability = (slope*raw + bias).sigmoid()
        eligible = data['usable'] & has_relation.bool()
        accepted = eligible & (probability >= threshold) & (threshold < 1.)
        if not enabled:
            accepted = torch.zeros_like(accepted)
        output = torch.where(accepted[:, None], proposal_logits.to(native_logits.dtype), native_logits)
        return dict(logits=output, proposal_logits=proposal_logits, accepted=accepted,
                    eligible=eligible, raw=raw, probability=probability,
                    visual_probability=data['visual_probability'])


def incremental_targets(native, proposal, grid_world, target_world, eligible):
    """TRAIN-only lexicographic improvement: R@1 first, then >=1m NE gain."""
    distance = (grid_world-target_world[:, None, :2]).norm(dim=-1)
    old = distance.gather(1, native.argmax(-1)[:, None]).squeeze(-1)
    new = distance.gather(1, proposal.argmax(-1)[:, None]).squeeze(-1)
    correct_old, correct_new = old <= 20, new <= 20
    better = (~correct_old & correct_new) | ((correct_old == correct_new) & (new <= old-1.))
    return dict(target=(better & eligible).float(), before_error=old,
                proposal_error=new, damaged=correct_old & ~correct_new,
                corrected=~correct_old & correct_new)


def fit_probability_calibration(raw, target):
    """Monotonic Platt calibration on disjoint TRAIN probability groups only."""
    raw, target = raw.detach().cpu().float(), target.detach().cpu().float()
    if not len(raw):
        return dict(slope=1., bias=-20., count=0, positives=0, class_complete=False)
    if not (target > 0).any() or not (target == 0).any():
        prior = (float(target.sum())+1)/(len(target)+2)
        return dict(slope=1e-4, bias=math.log(prior/(1-prior)), count=len(raw), positives=int(target.sum()), class_complete=False)
    log_slope = torch.zeros((), requires_grad=True)
    bias = torch.tensor(math.log((float(target.sum())+1)/(len(target)-float(target.sum())+1)), requires_grad=True)
    optimizer = torch.optim.Adam([log_slope, bias], lr=.05)
    for _ in range(200):
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(log_slope.exp()*raw+bias, target) + .001*log_slope.square()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            log_slope.clamp_(-5, 3)
            bias.clamp_(-15, 15)
    return dict(slope=float(log_slope.detach().exp()), bias=float(bias.detach()), count=len(raw), positives=int(target.sum()), class_complete=True)


def select_safe_threshold(probability, eligible, native_error, proposal_error, a0_error):
    """TRAIN-only selection with zero incremental damage vs Native and A0.

    The no-op is always feasible; zero TRAIN damage is not a Dev guarantee.
    """
    grid = sorted(set([1., .01, .025, .05, .1, .2, .3, .4, .5, .7, .9] +
                      probability[eligible].detach().cpu().tolist()))
    table = []
    for threshold in grid:
        accepted = eligible & (probability >= threshold) & (threshold < 1.)
        error = torch.where(accepted, proposal_error, native_error)
        gain = int(((native_error > 20) & (error <= 20)).sum())
        harm = int(((native_error <= 20) & (error > 20)).sum())
        a0_extra_harm = int(((a0_error <= 20) & (native_error <= 20) & (error > 20)).sum())
        # Also protect every A0-correct state already damaged by Native from further NE degradation.
        a0_ne_worse = int(((a0_error <= 20) & accepted & (error > native_error+1e-5)).sum())
        table.append(dict(threshold=float(threshold), accepted=int(accepted.sum()),
                          corrected=gain, damaged=harm, a0_extra_harm=a0_extra_harm,
                          a0_ne_worse=a0_ne_worse, net=gain-harm,
                          ne_gain=float((native_error-error).sum())))
    feasible = [r for r in table if r['damaged']==0 and r['a0_extra_harm']==0 and r['a0_ne_worse']==0]
    chosen = max(feasible, key=lambda r:(r['net'], r['ne_gain'], -r['accepted'], r['threshold']))
    return dict(selected=chosen, sweep=table, damage_budget=0)
