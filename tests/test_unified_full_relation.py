"""CPU-only contracts for the all-instruction CityNav integration."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from graphnav.full_localization import GoalField
from graphnav.unified_pgvb import UnifiedPGVB


class FakeGeometry(nn.Module):
    hidden = 128

    def forward(self, static, *, variant, tokens, token_mask, node_emb):
        assert variant == "geometry"
        logits = static["prior"]
        xy = static["grid_xy"]
        ctx = static["context"]
        ids = logits.topk(3, -1).indices
        return GoalField(
            belief_logits=logits,
            candidate_ids=ids,
            candidate_xy=xy.gather(1, ids[..., None].expand(-1, -1, 2)),
            candidate_context=ctx.gather(1, ids[..., None].expand(-1, -1, 128)),
            dense_context=ctx,
        )


def _inputs():
    torch.manual_seed(42)
    b, q, refs = 2, 9, 3
    prior = torch.arange(q, dtype=torch.float32)[None].expand(b, -1).clone()
    grid = torch.rand(b, q, 2)
    context = torch.randn(b, q, 128)
    pair = torch.randn(b, q, refs, 10)
    return dict(
        static_inputs=dict(prior=prior, grid_xy=grid, context=context,
                           pair_geometry=pair,
                           entity_valid=torch.tensor([[1, 0, 1], [0, 0, 0]],
                                                     dtype=torch.bool)),
        tokens=torch.randn(b, 5, 768),
        token_mask=torch.ones(b, 5, dtype=torch.bool),
        node_emb=torch.randn(b, refs, 768),
    )


def test_untrained_full_relation_preserves_geometry_with_or_without_anchor():
    model = UnifiedPGVB(geometry=FakeGeometry(), full_relation=True).train()
    kwargs = _inputs()
    result = model.forward_full_relation(**kwargs)
    base = result["geometry_field"]
    torch.testing.assert_close(result["field"].belief_logits, base.belief_logits)
    torch.testing.assert_close(result["relation_field"].belief_logits, base.belief_logits)
    assert result["field"].candidate_ids.shape == (2, 3)
    assert result["relation_delta"].shape == (2, 9)
    assert not result["audit"]["active"]


def test_trained_relation_scores_nonempty_reference_but_not_empty():
    model = UnifiedPGVB(geometry=FakeGeometry(), full_relation=True).train()
    inputs = _inputs()
    with torch.no_grad():
        model.full_relation.score.weight.fill_(0.12)
    result = model.forward_full_relation(**inputs)
    assert torch.equal(result["relation_delta"][1], torch.zeros(9))
    assert torch.isfinite(result["field"].belief_logits).all()
    torch.testing.assert_close(
        result["field"].candidate_xy,
        inputs["static_inputs"]["grid_xy"].gather(
            1, result["field"].candidate_ids[..., None].expand(-1, -1, 2),
        ),
    )
    assert result["relation_delta"][0].abs().max() > 0


def test_mixed_anchor_gradient_preserves_full_text_token_path():
    model = UnifiedPGVB(geometry=FakeGeometry(), full_relation=True).train()
    args = _inputs()
    result = model.forward_full_relation(**args)
    loss = result["field"].belief_logits.sum()
    loss.backward()
    assert model.full_relation.score.weight.grad is not None
    assert model.full_relation.score.weight.grad.abs().sum() > 0


def test_evaluation_requires_trained_explicit_checkpoint_flag():
    model = UnifiedPGVB(geometry=FakeGeometry(), full_relation=True).eval()
    with pytest.raises(RuntimeError, match="Untrained"):
        model.forward_full_relation(**_inputs())
    model.full_relation_ready = True
    assert model.forward_full_relation(**_inputs())["field"].candidate_ids.shape == (2, 3)


def test_public_landmark_name_embeddings_are_actually_used():
    model = UnifiedPGVB(geometry=FakeGeometry(), full_relation=True).train()
    inputs = _inputs()
    with torch.no_grad():
        model.full_relation.score.weight.fill_(0.12)
    original = model.forward_full_relation(**inputs)["relation_delta"]
    changed = dict(inputs)
    changed["node_emb"] = inputs["node_emb"].clone()
    changed["node_emb"][0, 0] += 5.0
    new = model.forward_full_relation(**changed)["relation_delta"]
    assert not torch.allclose(original[0], new[0])
    # No valid public references: original name changes cannot alter Belief.
    assert torch.equal(original[1], new[1])


def test_original_geometry_baseline_cannot_mix_trained_relation_or_rgb():
    from graphnav.unified_full_relation_runtime import UnifiedPGVBFullRelationPolicy
    with pytest.raises(ValueError,match='isolated frozen baseline'):
        UnifiedPGVBFullRelationPolicy(None,None,None,None,geometry_source='original',relation_control='full',visual_control='off')
    with pytest.raises(ValueError,match='isolated frozen baseline'):
        UnifiedPGVBFullRelationPolicy(None,None,None,None,geometry_source='original',relation_control='off',visual_control='tracked')
