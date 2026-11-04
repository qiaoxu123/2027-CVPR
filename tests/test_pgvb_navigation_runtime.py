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


def test_visual_reorder_uses_same_world_grid_and_exact_dense_context():
    field, uv = _original()
    updated, audit = bounded_pgvb_field(field, uv, uv[:, :4],
                                      torch.tensor([[-8., -4., 2., 10.]]),
                                      torch.ones(1, 4, dtype=torch.bool))
    assert audit['active']
    assert torch.equal(updated.candidate_xy, uv.gather(1, updated.candidate_ids[..., None].expand(-1, -1, 2)))
    assert torch.equal(updated.candidate_context, field.dense_context.gather(
        1, updated.candidate_ids[..., None].expand(-1, -1, field.dense_context.shape[-1])))


def test_pack_pgvb_audit_is_detached_and_does_not_change_live_field():
    from graphnav.pgvb_closedloop_audit import pack_field
    field, _ = _original()
    before = field.belief_logits.clone()
    packed = pack_field(field)
    packed['logits'][0] = 99
    assert torch.equal(field.belief_logits, before)
    assert packed['candidate_context'].shape == (8, 16)


@__import__('pytest').mark.parametrize('key', [666, '666'])
def test_named_catalog_accepts_integer_or_string_public_object_ids(monkeypatch,key):
    from types import SimpleNamespace
    from graphnav.pgvb_navigation_runtime import PGVBNamedLandmarkPolicy
    from graphnav.stage1_runtime import GraphNavStage1Policy
    monkeypatch.setattr(GraphNavStage1Policy,'set_episode',lambda *a,**k:None)
    policy=object.__new__(PGVBNamedLandmarkPolicy)
    episode=SimpleNamespace(reference_names=('Example Road',),map_name='example')
    catalog=SimpleNamespace(objects={'example':{key:dict(name='Example Road',object_type='TrafficRoad',contour=[[0,0],[10,0],[10,2],[0,2]])}})
    policy.set_episode(episode,None,catalog)
    assert policy._landmark['typ']=='Road'
    assert len(policy._landmark['contours'])==1


def test_standalone_no_view_holds_xyz_without_leaking_yaw_into_waypoint():
    import numpy as np
    from types import SimpleNamespace
    from graphnav.pgvb_navigation_runtime import PGVBStandaloneNamedPolicy
    p=object.__new__(PGVBStandaloneNamedPolicy)
    p.record={'instruction':'Navigate to Example.'};p.observed_steps=0;p.audit_callback=None;p.pgvb_seconds=0.
    p._timed_scores=lambda observation:(None,'no_supported_local_anchor')
    state=SimpleNamespace(pose=np.array([1.,2.,30.,1.57]),current_observation={})
    result=p.predict(state,p.record['instruction'])
    assert result.xyz.shape==(3,)
    np.testing.assert_equal(result.xyz,state.pose[:3])
    assert not result.should_stop
    assert result.diagnostics['abstained_no_visual_goal']
    assert result.diagnostics['geometry_branch_calls']==0
