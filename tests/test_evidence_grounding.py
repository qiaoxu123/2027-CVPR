"""Regression tests for opt-in relation and observed-only visual refinement."""
import pytest
import torch

from graphnav.full_localization import GoalField
from graphnav.evidence_grounding import (
    RasterizedEvidence, SpatialEvidenceMemory, TargetEvidenceRefiner,
    appearance_query, rasterize_observed_patches,
)


def field():
    torch.manual_seed(19)
    dense = torch.randn(1, 16, 8)
    grid = torch.stack(torch.meshgrid(
        torch.linspace(0.1, .9, 4), torch.linspace(0.1, .9, 4),
        indexing="ij",
    ), -1).reshape(1, 16, 2)
    logits = torch.randn(1, 16)
    ids = logits.topk(3, -1).indices
    return GoalField(
        logits, ids, grid.gather(1, ids[..., None].expand(-1, -1, 2)),
        dense.gather(1, ids[..., None].expand(-1, -1, 8)),
        dense_context=dense,
    ), grid


def test_zero_adapter_preserves_a3_bitwise():
    base, grid = field()
    adapter = TargetEvidenceRefiner(hidden=8, text_dim=12, vision_dim=6, relation_dim=5)
    out = adapter(
        base, grid_xy=grid, relation_fields=torch.randn(1, 16, 5),
        text_tokens=torch.randn(1, 7, 12),
        text_mask=torch.ones(1, 7, dtype=torch.bool),
        patch_features=torch.randn(1, 4, 6),
        patch_xy=grid[:, :4].clone(),
        patch_valid=torch.ones(1, 4, dtype=torch.bool),
    )
    assert torch.equal(out.belief_logits, base.belief_logits)
    assert torch.equal(out.candidate_ids, base.candidate_ids)
    out.belief_logits.sum().backward()
    assert adapter.relation_gate.grad is not None
    assert adapter.visual_gate.grad is not None


def test_unobserved_never_changes_with_positive_gate():
    base, grid = field()
    adapter = TargetEvidenceRefiner(
        hidden=8, text_dim=12, vision_dim=6, relation_dim=5, sigma=.035
    )
    with torch.no_grad():
        adapter.visual_gate.fill_(1.)
    patch_xy = grid[:, :1].clone()
    result = adapter(
        base, grid_xy=grid, text_tokens=torch.randn(1, 3, 12),
        text_mask=torch.ones(1, 3, dtype=torch.bool),
        patch_features=torch.randn(1, 1, 6),
        patch_xy=patch_xy, patch_valid=torch.tensor([[True]]),
    )
    # The four-by-four grid has 0.2667 spacing, outside raster support.
    assert torch.equal(result.belief_logits[:, 1:], base.belief_logits[:, 1:])
    unseen = adapter(
        base, grid_xy=grid, text_tokens=torch.randn(1, 3, 12),
        text_mask=torch.ones(1, 3, dtype=torch.bool),
        patch_features=torch.randn(1, 1, 6),
        patch_xy=patch_xy, patch_valid=torch.tensor([[False]]),
    )
    assert torch.equal(unseen.belief_logits, base.belief_logits)


def test_geometry_support_is_sparse_and_masked():
    _, grid = field()
    points = grid[:, :2].clone()
    score = torch.tensor([[1., -1.]])
    valid = torch.tensor([[True, False]])
    evidence = rasterize_observed_patches(score, points, valid, grid, sigma=.02)
    assert evidence.coverage[0, 0] > 0
    assert evidence.coverage[0, 1] == 0
    assert evidence.scores[0, 0] > 0
    assert torch.count_nonzero(evidence.coverage) == 1


def test_history_deduplicates_same_view_and_rejects_noncausal():
    memory = SpatialEvidenceMemory(1, 2, "cpu", min_move_m=2.)
    a = RasterizedEvidence(torch.tensor([[1., 2.]]), torch.ones(1, 2))
    b = RasterizedEvidence(torch.tensor([[-1., -2.]]), torch.ones(1, 2))
    pose = torch.tensor([[5., 10., .0]])
    memory.update(a, step=0, pose_xyyaw=pose)
    old = memory.snapshot()
    memory.update(b, step=1, pose_xyyaw=pose)
    assert torch.equal(old.scores, memory.snapshot().scores)
    assert torch.equal(old.coverage, memory.snapshot().coverage)
    memory.update(b, step=2, pose_xyyaw=torch.tensor([[8., 10., 0.]]))
    assert torch.equal(memory.snapshot().scores, torch.zeros(1, 2))
    with pytest.raises(ValueError, match="advance"):
        memory.update(a, step=2, pose_xyyaw=pose)


def test_real_descent_updates_memory_but_yaw_only_does_not():
    memory = SpatialEvidenceMemory(1, 2, "cpu")
    first = RasterizedEvidence(torch.ones(1, 2), torch.ones(1, 2))
    sharper = RasterizedEvidence(torch.full((1, 2), 3.), torch.ones(1, 2))
    memory.update(first, step=0, pose_xyyaw=torch.tensor([[5., 10., 80., 0.]]))
    memory.update(sharper, step=1, pose_xyyaw=torch.tensor([[5., 10., 80., 1.]]))
    assert torch.equal(memory.snapshot().scores, torch.ones(1, 2))
    memory.update(sharper, step=2, pose_xyyaw=torch.tensor([[5., 10., 70., 1.]]))
    assert torch.equal(memory.snapshot().scores, torch.full((1, 2), 2.))


def test_kernel_cannot_update_grid_outside_actual_observation():
    base, grid = field()
    adapter = TargetEvidenceRefiner(hidden=8, text_dim=12, vision_dim=6,
                                    relation_dim=5, sigma=1.)
    with torch.no_grad():
        adapter.visual_gate.fill_(1.)
    support = torch.zeros(1, 16, dtype=torch.bool)
    support[:, 0] = True
    memory = SpatialEvidenceMemory(1, 16, "cpu")
    out = adapter(base, grid_xy=grid, text_tokens=torch.randn(1, 3, 12),
                  text_mask=torch.ones(1, 3, dtype=torch.bool),
                  patch_features=torch.randn(1, 1, 6), patch_xy=grid[:, :1],
                  patch_valid=torch.ones(1, 1, dtype=torch.bool),
                  observed_grid_valid=support, memory=memory, step=0,
                  pose_xyyaw=torch.tensor([[5., 10., 80., 0.]]))
    assert torch.equal(out.belief_logits[:, 1:], base.belief_logits[:, 1:])
    assert torch.count_nonzero(memory.weights[:, 1:]) == 0


def test_relation_gate_and_dense_fullmap_update():
    base, grid = field()
    adapter = TargetEvidenceRefiner(hidden=8, text_dim=12, vision_dim=6, relation_dim=5)
    with torch.no_grad():
        adapter.relation_gate.fill_(1.)
        adapter.relation[-1].weight.zero_()
        adapter.relation[-1].bias.fill_(2)
    relation = torch.zeros(1, 16, 5)
    relation[:, 12:] = 1
    changed = adapter(base, grid_xy=grid, relation_fields=relation)
    assert torch.equal(changed.belief_logits[:, :12], base.belief_logits[:, :12])
    assert (changed.belief_logits[:, 12:] > base.belief_logits[:, 12:]).all()


def test_strict_shapes_and_no_noncausal_stuff():
    base, grid = field()
    adapter = TargetEvidenceRefiner(hidden=8, text_dim=12, vision_dim=6, relation_dim=5)
    with pytest.raises(ValueError, match="patch"):
        adapter(base, grid_xy=grid, patch_features=torch.zeros(1, 1, 6))
    with pytest.raises(ValueError, match="valid"):
        rasterize_observed_patches(
            torch.ones(1, 1), grid[:, :1],
            torch.ones(1, 1), grid,
        )
    with pytest.raises(ValueError, match="instruction"):
        appearance_query("")
    assert "brown" in appearance_query("The second brown house near Wellington Road").lower()
