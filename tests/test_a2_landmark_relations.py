"""A2 CPU checks: language reference encoding, geometry and conservative belief."""
import numpy as np
import torch

from experiments.a2_waldo_landmark_belief.core import (
    BoxLandmarkRelationScorer,
    InstructionLandmarkEncoder,
    box_landmark_geometry,
    conservative_belief_mix,
)


def rect(cx, cy, half=2):
    return np.array([
        [cx-half, cy-half], [cx+half, cy-half],
        [cx+half, cy+half], [cx-half, cy+half],
    ], np.float32)


def test_every_box_times_landmark_geometric_orientation_and_height_mask():
    boxes = [rect(20, 0), rect(-20, 0), rect(0, 15)]
    refs = [rect(0, 0, 5), rect(0, 30, 5)]
    features, valid = box_landmark_geometry(
        boxes, refs, [.9, .8, .7],
        box_height_m=[10.0, None, 1.0],
        landmark_height_m=[4.0, None],
    )
    assert features.shape == (3, 2, 12)
    assert valid.all()
    assert features[0, 0, 0] > 0 and features[0, 0, 4] > .99
    assert features[1, 0, 0] < 0 and features[1, 0, 4] < -.99
    assert features[2, 0, 1] > 0 and features[2, 0, 3] > .99
    assert features[0, 0, 11] == 1 and features[0, 0, 10] > 0
    assert features[1, 0, 11] == 0 and features[1, 0, 10] == 0
    assert features[:, 1, 11].sum() == 0


def test_invalid_polygon_is_masked_not_negative():
    features, valid = box_landmark_geometry(
        [rect(2, 3)], [np.full((4, 2), np.nan)], [.9],
    )
    assert not valid.any()
    assert features.sum() == 0


def test_linguistic_mentions_encoded_separately_and_unlinked_refs_masked():
    torch.manual_seed(4)
    model = InstructionLandmarkEncoder(text_dim=8, hidden=16, heads=4)
    tokens = torch.randn(1, 6, 8)
    names = torch.randn(1, 3, 2, 8)
    mention = torch.zeros(1, 3, 6, dtype=torch.bool)
    mention[0, 0, 1] = True
    mention[0, 1, 4] = True
    mention[0, 2, 3] = True
    encoding, eligible = model(
        tokens, torch.ones(1, 6, dtype=torch.bool), mention,
        names, torch.ones(1, 3, 2, dtype=torch.bool),
        torch.tensor([[True, True, False]]),
    )
    assert encoding.shape == (1, 3, 16)
    assert eligible.tolist() == [[True, True, False]]
    assert not torch.allclose(encoding[0, 0], encoding[0, 1])
    assert encoding[0, 2].abs().sum() == 0


def test_pair_rank_uses_landmark_tokens_and_geometry_gradient():
    torch.manual_seed(5)
    encoder = InstructionLandmarkEncoder(text_dim=8, hidden=16, heads=4)
    scorer = BoxLandmarkRelationScorer(vision_dim=12, hidden=16)
    tokens = torch.randn(1, 6, 8)
    names = torch.randn(1, 2, 3, 8)
    mentions = torch.zeros(1, 2, 6, dtype=torch.bool)
    mentions[0, 0, 1] = True
    mentions[0, 1, 4] = True
    landmark, lv = encoder(
        tokens, torch.ones(1, 6, dtype=torch.bool), mentions,
        names, torch.ones(1, 2, 3, dtype=torch.bool),
        torch.tensor([[True, True]]),
    )
    boxes = torch.randn(1, 3, 12)
    geom = torch.randn(1, 3, 2, 12)
    rank, pairs, reliable = scorer(
        boxes, torch.tensor([[True, True, False]]),
        landmark, lv, geom, torch.ones(1, 3, 2, dtype=torch.bool),
    )
    assert rank.shape == (1, 3) and pairs.shape == (1, 3, 2)
    assert reliable.tolist() == [[True, True, False]]
    assert rank[0, 2].item() < -1e3
    rank[0, :2].sum().backward()
    assert scorer.geometry[0].weight.grad is not None
    assert scorer.geometry[0].weight.grad.abs().sum() > 0
    assert encoder.phrase_name[1].weight.grad is not None
    assert encoder.phrase_name[1].weight.grad.abs().sum() > 0


def test_belief_conserves_probability_and_outside_mass():
    original = torch.tensor([[2., 1., 0., -1., -2.]])
    supports = torch.tensor([[
        [False, True, True, False, False],
        [False, False, True, True, False],
    ]])
    scores = torch.tensor([[2.0, 0.5]])
    eligible = torch.tensor([[True, True]])
    output = conservative_belief_mix(original, scores, supports, eligible, alpha=.25)
    before = original.softmax(-1)
    after = output.softmax(-1)
    torch.testing.assert_close(after.sum(-1), torch.ones(1), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(after[:, [0, 4]], before[:, [0, 4]], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(after[:, 1:4].sum(-1), before[:, 1:4].sum(-1), rtol=1e-6, atol=1e-6)
    assert torch.equal(output[:, [0, 4]], original[:, [0, 4]])
    assert not torch.equal(output, original)


def test_default_gate_and_no_valid_candidates_are_exact_fallback():
    original = torch.randn(1, 8)
    scores = torch.randn(1, 2)
    support = torch.zeros(1, 2, 8, dtype=torch.bool)
    valid = torch.ones(1, 2, dtype=torch.bool)
    assert conservative_belief_mix(original, scores, support, valid) is original
    assert torch.equal(conservative_belief_mix(original, scores, support, valid, alpha=.2), original)
