"""Protect TRAIN-only fitting and identical fixed capacity in the CPU probe."""
import importlib

import pytest
import torch


def api():
    return importlib.import_module('diagnostics.evidence_quality.linear_fit')


def identities():
    return [dict(role='train', map_name='map', object_id=i) for i in range(12)]


def sample():
    x = torch.zeros(12, 22); x[:, 0] = torch.arange(12).float()
    y = torch.tensor([0, 0, 0, 1, 1, 2, 0, 2, 1, 1, 1, 2])
    return x, y


def test_group_split_is_fixed_and_keeps_repeated_targets_together():
    ids = identities() + identities()[5:6]
    fit = api().training_partition(ids)
    assert fit.tolist() == [True, True, True, True, True, False, True,
                            False, False, True, True, True, False]


def test_nontrain_role_rejected_before_reading_features():
    for role in ('dev', 'calibration_fit', 'calibration_verify', 'val_unseen'):
        with pytest.raises(ValueError):
            api().fit_linear(None, None, [dict(role=role, map_name='map', object_id=1)], width=15)


def test_selection_values_and_outcomes_cannot_affect_fitted_model():
    x, y = sample(); original = api().fit_linear(x, y, identities(), width=22)
    x[[5, 7, 8]] = 100000.; y[[5, 7, 8]] = 0
    changed = api().fit_linear(x, y, identities(), width=22)
    for key in ('mean', 'scale', 'weight', 'bias'):
        assert torch.equal(original[key], changed[key])
    assert original['history'] == changed['history']
    assert len(original['history']) == 20
    assert original['history'][-1]['loss'] < original['history'][0]['loss']


def test_adding_zero_features_preserves_reference_predictions():
    x, y = sample()
    baseline = api().fit_linear(x, y, identities(), width=15)
    extended = api().fit_linear(x, y, identities(), width=22)
    torch.testing.assert_close(api().predict(baseline, x), api().predict(extended, x),
                               atol=1e-7, rtol=1e-7)


def test_weighted_classes_are_corrected_at_probability_output():
    state = dict(width=15, mean=torch.zeros(15), scale=torch.ones(15),
                 weight=torch.zeros(3, 15), bias=torch.zeros(3))
    result = api().predict(state, torch.zeros(2, 22))
    torch.testing.assert_close(result, torch.tensor([[24/31, 3/31, 4/31]]).expand(2, -1))


def test_no_selection_groups_is_reported_instead_of_fabricated_split():
    with pytest.raises(ValueError):
        api().fit_linear(torch.zeros(1, 22), torch.zeros(1, dtype=torch.long),
                         identities()[:1], width=15)


@pytest.mark.parametrize('change', ['nonfinite', 'bad_outcome', 'wrong_width'])
def test_invalid_fit_input_rejected(change):
    x, y = sample(); width = 15
    if change == 'nonfinite': x[0, 0] = float('inf')
    if change == 'bad_outcome': y[0] = 3
    if change == 'wrong_width': width = 16
    with pytest.raises(ValueError):
        api().fit_linear(x, y, identities(), width=width)


def test_every_epoch_retains_its_actual_weights_and_final_matches_epoch20():
    x,y=sample();state=api().fit_linear(x,y,identities(),width=22)
    assert [r['epoch'] for r in state['snapshots']]==list(range(1,21))
    assert torch.equal(state['weight'],state['snapshots'][-1]['weight'])
    assert torch.equal(state['bias'],state['snapshots'][-1]['bias'])
    assert not torch.equal(state['snapshots'][0]['weight'],state['snapshots'][-1]['weight'])
