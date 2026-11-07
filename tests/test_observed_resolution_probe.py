import torch
from scripts.probe_observed_roi_resolution import public_choices


def test_missing_top1_evidence_preserves_geometry_for_every_arm():
    valid = torch.tensor([[False, False], [True, True], [True, False]])
    choices, _ = public_choices(torch.randn(2, 3, 2, 4), valid,
                                torch.randn(2, 4), 42)
    assert set(choices.values()) == {0}


def test_same_resolution_features_give_same_paired_decisions():
    valid = torch.tensor([[True, True], [True, True], [False, False]])
    features = torch.randn(1, 3, 2, 4).expand(2, -1, -1, -1).clone()
    features[:, 2] = 1000  # unsupported content must never become a choice
    choices, _ = public_choices(features, valid, torch.randn(2, 4), 12)
    assert 2 not in choices.values()
    for channel in ('full', 'appearance', 'full_shuffle', 'appearance_shuffle'):
        assert choices['current384_'+channel] == choices['native_roi_'+channel]
