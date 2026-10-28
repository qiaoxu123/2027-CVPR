"""Catch invalid/padded evidence contaminating the diagnostic's public features."""
import importlib
import math

import pytest
import torch


def api():
    return importlib.import_module('diagnostics.evidence_quality.public_features')


def evidence():
    geometry = torch.zeros(1, 3, 3, 14)
    geometry[0, 0, 0, [9, 13, 11]] = torch.tensor([.8, .4, 1.])
    geometry[0, 1, 0, [9, 13, 11]] = torch.tensor([.2, .6, 0.])
    geometry[0, 1, 1, [9, 13, 11]] = torch.tensor([.6, .8, 1.])
    return dict(
        base=torch.zeros(1, 2), grid=torch.tensor([[[0., 0.], [10., 0.]]]),
        candidate_xy=torch.tensor([[[0., 0.], [10., 0.], [50., 0.]]]),
        candidate_support=torch.tensor([[[True, False], [False, True], [False, False]]]),
        scores=torch.zeros(1, 3), match_logits=torch.zeros(1, 3),
        reliable=torch.ones(1, 3, dtype=torch.bool),
        box_valid=torch.ones(1, 3, dtype=torch.bool),
        reference_valid=torch.tensor([[True, True, False]]),
        pair_valid=torch.tensor([[[True, False, True], [True, True, False], [True, True, True]]]),
        pair_geometry=geometry)


def test_excludes_unsupported_boxes_and_invalid_reference_from_weighted_quality():
    # Counting all reliable boxes, or averaging padded references, breaks this.
    x = api().public_features(**evidence())
    assert x.shape == (1, 22)
    assert x[0, 15:].tolist() == pytest.approx(
        [math.log(3), math.log(3), 1., 0., .6, .55, .75])


def test_missing_candidates_produce_no_extra_evidence():
    e = evidence(); e['candidate_support'].zero_()
    x = api().public_features(**e)
    assert torch.isfinite(x).all()
    assert torch.equal(x[:, 15:], torch.zeros(1, 7))
    assert x[0, 14] == 0


def test_single_candidate_has_no_competition():
    e = evidence(); e['candidate_support'][0, 1].zero_()
    x = api().public_features(**e)
    assert x[0, 15:].tolist() == pytest.approx(
        [math.log(2), math.log(3), 0., 1., .8, .4, 1.])


def test_masked_geometry_and_unsupported_scores_do_not_change_features():
    e = evidence(); original = api().public_features(**e)
    e['pair_geometry'][0, 0, 1:] = 123.
    e['pair_geometry'][0, 1, 2] = -321.
    e['pair_geometry'][0, 2] = 999.
    e['scores'][0, 2] = 1000.
    assert torch.equal(api().public_features(**e), original)


def test_target_or_outcome_cannot_enter_public_feature_interface():
    with pytest.raises(TypeError):
        api().public_features(**evidence(), target_xy=torch.zeros(1, 2))


@pytest.mark.parametrize('field', ['scores', 'pair_geometry', 'base'])
def test_nonfinite_public_input_rejected(field):
    e = evidence(); e[field].flatten()[0] = float('nan')
    with pytest.raises(ValueError):
        api().public_features(**e)


def test_axis_mismatch_rejected():
    e = evidence(); e['reference_valid'] = torch.ones(1, 2, dtype=torch.bool)
    with pytest.raises(ValueError):
        api().public_features(**e)


def test_match_confidence_changes_candidate_competition_weights():
    # Ignoring match logits would incorrectly keep the two weights equal.
    e = evidence(); e['match_logits'][0, :2] = torch.tensor([math.log(3), -math.log(3)])
    x = api().public_features(**e)
    assert x[0, 17:].tolist() == pytest.approx([.8112781245, .5, .7, .475, .875])


def test_actual_one_box_axis_has_finite_unit_margin():
    e = evidence()
    for name in ('candidate_xy', 'candidate_support', 'scores', 'match_logits',
                 'reliable', 'box_valid', 'pair_valid', 'pair_geometry'):
        e[name] = e[name][:, :1]
    x = api().public_features(**e)
    assert x[0, 17:].tolist() == pytest.approx([0., 1., .8, .4, 1.])
