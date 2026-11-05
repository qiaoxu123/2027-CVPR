"""CPU regression tests for the original CityNav-native RAGM-inspired head."""
import torch
import pytest
from graphnav.aerialvg_inspired_relation import (
    CandidateAnchorRelation, apply_relation_residual,
)


def _inputs():
    torch.manual_seed(4)
    b, q, r, l = 2, 9, 3, 5
    dense = torch.randn(b, q, 16)
    tokens = torch.randn(b, l, 32)
    text_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool)
    geometry = torch.randn(b, r, q, 10)
    ids = torch.tensor([[0, 1, 2], [2, 1, 0]])
    valid = torch.tensor([[1, 1, 0], [1, 0, 1]], dtype=torch.bool)
    return dense, tokens, text_mask, geometry, ids, valid


def _head():
    return CandidateAnchorRelation(
        context_dim=16, text_dim=32, hidden=24, heads=4,
        chunk_size=4, max_logit_delta=1.0,
    ).eval()


def test_untrained_head_exactly_preserves_geometry():
    head = _head()
    inputs = _inputs()
    residual = head(*inputs)
    assert residual.shape == (2, 9)
    assert torch.equal(residual, torch.zeros_like(residual))
    base = torch.randn(2, 9)
    torch.testing.assert_close(apply_relation_residual(base, residual), base)


def test_anchor_order_invariant():
    head = _head()
    with torch.no_grad():
        head.score.weight.fill_(0.1)
    data = _inputs()
    original = head(*data)
    order = torch.tensor([2, 0, 1])
    reordered = head(data[0], data[1], data[2],
                     data[3][:, order], data[4][:, order], data[5][:, order])
    torch.testing.assert_close(original, reordered, atol=1e-5, rtol=1e-5)


def test_invalid_anchor_does_not_change_prediction():
    head = _head()
    with torch.no_grad():
        head.score.weight.fill_(0.1)
    data = list(_inputs())
    original = head(*data)
    mask = ~data[5]
    corrupted = data[3].clone()
    corrupted[mask] += 10000
    data[3] = corrupted
    torch.testing.assert_close(original, head(*data), atol=1e-5, rtol=1e-5)


def test_gradients_flow_to_relation_weights():
    head = _head().train()
    data = _inputs()
    loss = head(*data).square().mean() + head(*data).sum()
    loss.backward()
    assert head.score.weight.grad is not None
    assert torch.isfinite(head.score.weight.grad).all()
    assert head.score.weight.grad.abs().sum() > 0


def test_zero_or_missing_text_or_anchor_rejected():
    head = _head()
    data = list(_inputs())
    data[2] = torch.zeros_like(data[2])
    with pytest.raises(ValueError, match="at least one"):
        head(*data)
    data = list(_inputs())
    data[5] = torch.zeros_like(data[5])
    with pytest.raises(ValueError, match="at least one"):
        head(*data)


def test_geometry_shape_guard():
    head = _head()
    data = list(_inputs())
    data[3] = data[3][:, :, :-1]
    with pytest.raises(ValueError, match="Misaligned"):
        head(*data)
