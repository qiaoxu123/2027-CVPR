"""CPU-only gate safety checks. Native model weights and GPU caches not needed."""
import inspect

import pytest
import torch
from torch.nn import functional as F

from graphnav.gated_belief_fusion import (
    FEATURE_NAMES, ReliabilityGatedBelief, count_outcomes,
    training_gate_targets, visual_hypothesis,
)
from scripts.train_native_gated_fusion import (
    select_threshold, split_train_by_group,
)


def sample():
    # The frozen A0 is very confident in 0m but the observed visual query
    # supports 50m. This case is impossible for a small clipped logit delta.
    base = torch.tensor([[6., 0., -2.]])
    query = torch.tensor([[5., 2.]])
    null = torch.tensor([0.])
    kernel = torch.tensor([[[0., 0.], [1., 0.], [0., 0.]]])
    observed = torch.tensor([[True, True, True]])
    valid = torch.tensor([[True, False]])
    geometry = torch.zeros(1, 2, 8)
    geometry[..., 2] = 1.
    grid = torch.tensor([[[0., 0.], [50., 0.], [100., 0.]]])
    return base, query, null, kernel, observed, valid, geometry, grid


def run(gate, *, enabled, threshold=.8, transfer=.9, values=None):
    values = sample() if values is None else values
    return gate(*values, enabled=enabled, threshold=threshold,
                transfer=transfer)


def test_default_is_exact_a0_even_when_visual_is_contrary():
    gate = ReliabilityGatedBelief()
    old = sample()[0]
    result = run(gate, enabled=False)
    assert torch.equal(result['logits'], old)
    assert not result['accepted'].any()
    assert len(FEATURE_NAMES) == 16


def test_untrained_gate_is_conservative():
    gate = ReliabilityGatedBelief()
    result = run(gate, enabled=True)
    assert not result['accepted'].any()
    assert torch.equal(result['logits'], sample()[0])


def test_reliable_gate_can_move_global_peak_without_logit_residual_bound():
    gate = ReliabilityGatedBelief()
    with torch.no_grad():
        gate.reliability_net[-1].bias.fill_(10.)
    result = run(gate, enabled=True)
    assert result['accepted'].item()
    assert result['logits'].argmax(-1).item() == 1
    assert torch.allclose(result['logits'].softmax(-1).sum(-1),
                          torch.ones(1), atol=1e-6)
    assert result['visual_probability'][0, 1].item() > .99
    assert result['attention'][0, 0].item() > .99


def test_no_supported_query_exact_a0_fallback():
    gate = ReliabilityGatedBelief()
    with torch.no_grad():
        gate.reliability_net[-1].bias.fill_(10.)
    values = list(sample())
    values[5] = torch.zeros_like(values[5])  # candidate validity
    result = run(gate, enabled=True, values=values)
    assert not result['accepted'].any()
    assert torch.equal(result['logits'], values[0])
    assert result['visual_probability'].sum().item() == 0.
    values = list(sample())
    values[4] = torch.zeros_like(values[4])  # observable support
    result = run(gate, enabled=True, values=values)
    assert not result['accepted'].any()
    assert torch.equal(result['logits'], values[0])


def test_incomplete_query_support_never_invents_unobserved_spatial_mass():
    values = list(sample())
    values[4] = torch.tensor([[False, True, False]])
    data = visual_hypothesis(*values)
    assert data['usable'].item()
    assert data['visual_probability'][0, 0].item() == 0.
    assert data['visual_probability'][0, 2].item() == 0.
    assert data['visual_probability'][0, 1].item() == 1.


def test_gt_is_not_part_of_gate_forward_but_trains_safe_action():
    assert 'target_world' not in inspect.signature(
        ReliabilityGatedBelief.forward).parameters
    vals = sample()
    data = visual_hypothesis(*vals)
    target = torch.tensor([[50., 0.]])
    labels = training_gate_targets(
        vals[0], data['visual_probability'], data['usable'], vals[-1],
        target, transfer=.9)
    assert labels['target'].item() == 1.
    gate = ReliabilityGatedBelief()
    result = run(gate, enabled=False)
    loss = F.binary_cross_entropy_with_logits(
        result['reliability_logit'], labels['target'])
    loss.backward()
    assert gate.reliability_net[-1].weight.grad is not None
    assert gate.reliability_net[-1].weight.grad.abs().sum().item() > 0


def test_unreachable_counterfactual_is_not_positive_training_label():
    vals = list(sample())
    vals[0] = torch.tensor([[30., 0., 0.]])
    data = visual_hypothesis(*vals)
    labels = training_gate_targets(
        vals[0], data['visual_probability'], data['usable'], vals[-1],
        torch.tensor([[50., 0.]]), transfer=.01)
    assert labels['visual_correct'].item()
    assert not labels['counterfactual_correct'].item()
    assert labels['target'].item() == 0.


def test_switch_preserves_correct_prior_when_visual_wrong():
    vals = sample()
    data = visual_hypothesis(*vals)
    labels = training_gate_targets(
        vals[0], data['visual_probability'], data['usable'], vals[-1],
        torch.tensor([[0., 0.]]), transfer=.9)
    assert labels['before'].item()
    assert labels['target'].item() == 0.


def test_outcome_counter_counts_correct_and_damaged():
    vals = sample()
    gate = ReliabilityGatedBelief()
    with torch.no_grad():
        gate.reliability_net[-1].bias.fill_(10.)
    result = run(gate, enabled=True)
    counts = count_outcomes(vals[0], result['logits'], vals[-1],
                           torch.tensor([[50., 0.]]))
    assert counts['corrected'] == 1
    assert counts['damaged'] == 0


def test_bad_shapes_and_nonfinite_values_fail_closed():
    vals = list(sample())
    vals[3] = torch.zeros(1, 3, 3)
    with pytest.raises(ValueError):
        visual_hypothesis(*vals)
    vals = list(sample())
    vals[0] = torch.tensor([[float('nan'), 0., 0.]])
    with pytest.raises(ValueError):
        visual_hypothesis(*vals)


def test_train_group_split_and_calibration_harm_constraint():
    groups = [{'group': 'map_one', 'key': i} for i in range(20)]
    groups += [{'group': 'map_two', 'key': i} for i in range(20)]
    groups += [{'group': 'map_three', 'key': i} for i in range(20)]
    groups += [{'group': 'map_four', 'key': i} for i in range(20)]
    groups += [{'group': 'map_five', 'key': i} for i in range(20)]
    groups += [{'group': 'map_six', 'key': i} for i in range(20)]
    fit, calib = split_train_by_group(groups)
    assert not {r['group'] for r in fit} & {r['group'] for r in calib}
    row = [
        dict(count=100, gate_threshold=.5, net_corrections=8, damaged=8, gate_accepted=25),
        dict(count=100, gate_threshold=.9, net_corrections=3, damaged=1, gate_accepted=7),
        dict(count=100, gate_threshold=1., net_corrections=0, damaged=0, gate_accepted=0),
    ]
    assert select_threshold(row, .02)['gate_threshold'] == .9


def test_no_op_threshold_one_is_always_exact_fallback():
    gate = ReliabilityGatedBelief()
    with torch.no_grad():
        gate.reliability_net[-1].bias.fill_(100.)
    result = run(gate, enabled=True, threshold=1.)
    assert not result['accepted'].any()
    assert torch.equal(result['logits'], sample()[0])


def test_at_zero_visual_support_reliability_never_causes_switch():
    vals = list(sample())
    vals[3] = torch.zeros_like(vals[3])
    gate = ReliabilityGatedBelief()
    with torch.no_grad():
        gate.reliability_net[-1].bias.fill_(100.)
    outcome = run(gate, enabled=True, values=vals, threshold=.5)
    assert not outcome['accepted'].any()
    assert torch.equal(outcome['logits'], vals[0])
