"""PGVB opt-in bridge invariants. Run with pytest in the server environment."""
import torch

from graphnav.full_localization import GoalField
from graphnav.pgvb_navigation_runtime import bounded_pgvb_field


def _original():
    n = 28 * 28
    uv = torch.stack(torch.meshgrid(
        (torch.arange(28).float() + 0.5) / 28,
        (torch.arange(28).float() + 0.5) / 28,
        indexing="ij"), -1).reshape(1, n, 2).flip(-1)
    logits = torch.zeros((1, n))
    ids = torch.arange(8).view(1, 8)
    context = torch.randn(1, n, 16, generator=torch.Generator().manual_seed(5))
    field = GoalField(logits, ids, uv.gather(1, ids[..., None].expand(-1, -1, 2)),
                      context.gather(1, ids[..., None].expand(-1, -1, 16)),
                      dense_context=context)
    return field, uv


def test_no_observation_is_identity():
    field, uv = _original()
    result, audit = bounded_pgvb_field(
        field, uv, torch.rand(1, 144, 2), torch.zeros(1, 144),
        torch.zeros(1, 144, dtype=torch.bool))
    assert result is field
    assert not audit["active"]


def test_constant_visual_scores_do_not_change_original_goal():
    field, uv = _original()
    coords = torch.rand(1, 144, 2)
    result, audit = bounded_pgvb_field(
        field, uv, coords, torch.ones(1, 144) * 7,
        torch.ones(1, 144, dtype=torch.bool))
    assert result is field
    assert not audit["active"]


def test_valid_visual_scores_update_only_covered_cells_and_keep_context():
    field, uv = _original()
    local_uv = torch.tensor([[[.26, .3], [.28, .3], [.29, .3], [.8, .8]]])
    scores = torch.tensor([[1.0, 3.0, 5.0, -5.0]])
    original = field.belief_logits.clone()
    updated, audit = bounded_pgvb_field(field, uv, local_uv, scores,
                                      torch.ones_like(scores, dtype=torch.bool))
    assert audit["active"]
    assert 0 < audit["covered_cells"] < 784
    assert audit["max_abs_delta"] <= 0.5 + 1e-6
    assert updated.dense_context is field.dense_context
    assert updated.candidate_context.shape == field.candidate_context.shape
    assert torch.equal(field.belief_logits, original)
    assert torch.equal(updated.belief_logits[updated.visual_coverage == 0],
                       original[updated.visual_coverage == 0])


def test_invalid_bounds_rejected():
    field, uv = _original()
    with __import__("pytest").raises(ValueError):
        bounded_pgvb_field(field, uv, torch.zeros(1, 4, 2),
                          torch.ones(1, 4), torch.ones(1, 4, dtype=torch.bool),
                          bound=1.5)
