"""CPU checks for dynamic-visual Belief and stable controller GoalField."""
import pytest
import torch

from graphnav.full_localization import GoalField
from graphnav.stable_navigation_goal import StableNavigationGoal, StableGoalConfig


def _field(values, ranked, *, covered=()):
    logits = torch.tensor([values], dtype=torch.float32)
    q = logits.shape[-1]
    xy = torch.stack((torch.arange(q) / q, torch.arange(q) / q), dim=-1)[None]
    context = torch.arange(q, dtype=torch.float32)[None, :, None].expand(1, q, 128).clone()
    ids = torch.tensor([ranked], dtype=torch.long)
    coverage = torch.zeros(1, q, dtype=torch.float32)
    coverage[0, list(covered)] = 1.
    return (
        GoalField(
            belief_logits=logits,
            candidate_ids=ids,
            candidate_xy=xy.gather(1, ids[..., None].expand(-1, -1, 2)),
            candidate_context=context.gather(1, ids[..., None].expand(-1, -1, 128)),
            dense_context=context,
            visual_coverage=coverage,
        ),
        xy,
    )


def _tracking():
    return dict(state="tracking", visual_gain=2.)


def test_hold_first_goal_across_changing_visual_top1_and_align_all_features():
    tracker = StableNavigationGoal()
    old, xy = _field([0, 0, 4, 2, 1, 0], [2, 3, 4])
    controlled, audit = tracker.select(old, xy)
    assert controlled is old and audit["event"] == "initialized"
    new, xy = _field([0, 0, 1, 5, 4, 0], [3, 4, 1], covered=[3, 4])
    held, audit = tracker.select(
        new, xy, tracking=_tracking(), visual_active=False,
    )
    assert audit["held_goal_id"] == 2
    assert audit["visual_top1_id"] == 3
    assert audit["event"] == "kept"
    assert held.candidate_ids.tolist() == [[2, 3, 4]]
    torch.testing.assert_close(held.candidate_xy[0, 0], xy[0, 2])
    torch.testing.assert_close(held.candidate_context[0, 0], new.dense_context[0, 2])
    assert held.belief_logits.data_ptr() == new.belief_logits.data_ptr()
    assert held.visual_coverage.data_ptr() == new.visual_coverage.data_ptr()
    assert new.candidate_ids.tolist() == [[3, 4, 1]]


def test_replanning_requires_two_identical_visual_supported_frames():
    tracker = StableNavigationGoal()
    base, xy = _field([0, 0, 4, 2, 0, 0], [2, 3, 4])
    tracker.select(base, xy)
    visual, xy = _field([0, 0, 2, 5, 0, 0], [3, 2, 4], covered=[3])
    one, state = tracker.select(visual, xy, tracking=_tracking(), visual_active=True)
    assert state["event"] == "pending"
    assert state["pending_count"] == 1
    assert one.candidate_ids[0, 0] == 2
    two, state = tracker.select(visual, xy, tracking=_tracking(), visual_active=True)
    assert state["event"] == "replanned"
    assert state["held_goal_id"] == 3
    assert state["replan_count"] == 1
    assert int(two.candidate_ids[0, 0]) == 3


def test_unobserved_competing_goal_never_replans():
    tracker = StableNavigationGoal()
    before, xy = _field([0, 0, 3, 0, 0, 0], [2, 3, 4])
    tracker.select(before, xy)
    visual, _ = _field([0, 0, 2, 5, 0, 0], [3, 2, 4], covered=[4])
    for _ in range(5):
        _, audit = tracker.select(visual, xy, tracking=_tracking(), visual_active=True)
    assert audit["held_goal_id"] == 2
    assert audit["pending_count"] == 0
    assert audit["replan_count"] == 0


def test_switching_visual_target_resets_continuous_confirmation():
    tracker = StableNavigationGoal()
    first, xy = _field([0, 0, 4, 0, 0, 0], [2, 3, 4])
    tracker.select(first, xy)
    a, _ = _field([0, 0, 1, 5, 0, 0], [3, 2, 4], covered=[3])
    b, _ = _field([0, 0, 1, 0, 5, 0], [4, 2, 3], covered=[4])
    tracker.select(a, xy, tracking=_tracking(), visual_active=True)
    _, state = tracker.select(b, xy, tracking=_tracking(), visual_active=True)
    assert state["pending_count"] == 1
    assert state["pending_goal_id"] == 4
    assert state["held_goal_id"] == 2


def test_tracking_loss_clears_pending_and_keeps_stable_goal():
    tracker = StableNavigationGoal()
    before, xy = _field([0, 0, 3, 0, 0, 0], [2, 3, 4])
    tracker.select(before, xy)
    after, _ = _field([0, 0, 1, 5, 0, 0], [3, 2, 4], covered=[3])
    tracker.select(after, xy, tracking=_tracking(), visual_active=True)
    _, lost = tracker.select(
        after, xy, tracking=dict(state="coasting", visual_gain=0), visual_active=False
    )
    assert lost["held_goal_id"] == 2
    assert lost["pending_count"] == 0
    _, reacquired = tracker.select(
        after, xy, tracking=_tracking(), visual_active=True
    )
    assert reacquired["pending_count"] == 1


def test_small_logit_margin_never_triggers_goal_change():
    tracker = StableNavigationGoal()
    base, xy = _field([0, 0, 4, 2, 0, 0], [2, 3, 4])
    tracker.select(base, xy)
    near_tie, _ = _field([0, 0, 4, 4.2, 0, 0], [3, 2, 4], covered=[3])
    for _ in range(4):
        _, audit = tracker.select(
            near_tie, xy, tracking=_tracking(), visual_active=True
        )
    assert audit["held_goal_id"] == 2
    assert audit["replan_count"] == 0


def test_episode_reset_isolation_and_full_candidate_context_consistency():
    tracker = StableNavigationGoal()
    original, xy = _field([0, 0, 3, 0, 0, 0], [2, 3, 4])
    tracker.select(original, xy)
    tracker.reset()
    updated, _ = _field([0, 0, 1, 5, 0, 0], [3, 2, 4], covered=[3])
    field, audit = tracker.select(updated, xy)
    assert audit["held_goal_id"] == 3
    assert audit["replan_count"] == 0
    assert audit["decisions"] == 1
    torch.testing.assert_close(field.candidate_context[0, 0], updated.dense_context[0, 3])


def test_bad_tensor_shapes_and_nonfinite_logits_are_rejected():
    tracker = StableNavigationGoal()
    field, xy = _field([0, 0, 3, 0, 0, 0], [2, 3, 4])
    with pytest.raises(ValueError, match="dense GoalField"):
        tracker.select(field, xy[:, :2])
    field.belief_logits[0, 3] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        tracker.select(field, xy)


def test_config_requires_repeated_evidence_and_valid_margin():
    with pytest.raises(ValueError):
        StableGoalConfig(replan_confirmations=1)
    with pytest.raises(ValueError):
        StableGoalConfig(replan_logit_margin=-1)


def test_real_runtime_wires_stabilized_candidate_into_frozen_controller(monkeypatch):
    from types import SimpleNamespace
    from graphnav.a3_runtime import A3TrajectoryPolicy
    from graphnav.unified_pgvb_runtime import UnifiedPGVBNamedLandmarkPolicy

    old, xy = _field([0, 0, 4, 2, 1, 0], [2, 3, 4])
    visual, _ = _field([0, 0, 1, 5, 4, 0], [3, 4, 1], covered=[3, 4])
    frames = [old, visual]
    seen = []

    class FakeUnified:
        direction_ready = False

        def forward_structured(self, spec, **kwargs):
            model_field = frames.pop(0)
            return {
                "geometry_field": old,
                "field": model_field,
                "evidence": None,
                "audit": dict(
                    active=(model_field is visual), covered_cells=2,
                    max_abs_delta=.5, changed_top1=(model_field is visual),
                    acquisition={
                        "state": "tracking" if model_field is visual else "geometry",
                        "visual_gain": 2. if model_field is visual else 0.,
                    },
                ),
            }

    obj = object.__new__(UnifiedPGVBNamedLandmarkPolicy)
    obj.record = {"instruction": "Navigate to Alpha Hall."}
    obj.mode = "tracked-stable"
    obj.device = torch.device("cpu")
    obj.inputs = {"grid_xy": xy}
    obj.tokens, obj.mask, obj.names = None, None, None
    obj.bound = .5
    obj._initialize_geometry_inputs = lambda: None
    obj._observed_inputs = lambda obs: (None, "synthetic_no_RGB_forward")
    obj._last_observation_metadata = None
    obj.stable_goal = StableNavigationGoal()
    obj.unified = FakeUnified()
    obj.audit_callback = lambda base, final, evidence: seen.append(
        int(final.candidate_ids[0, 0])
    )
    obj.pgvb_calls, obj.pgvb_seconds = 0, 0.
    obj.instruction_spec = {
        "version": "unified-instruction-v1",
        "intent": "navigate_to_landmark", "landmark": {"public_ids": ["a"]},
    }
    obj._public_region = None

    def check_controller(self, state, field):
        # The original frozen head receives the held Goal and fully aligned
        # candidate coordinate/context. The stop implementation is unchanged.
        seen.append(int(field.candidate_ids[0, 0]))
        torch.testing.assert_close(field.candidate_xy[0, 0], xy[0, 2])
        return SimpleNamespace(diagnostics={})

    monkeypatch.setattr(A3TrajectoryPolicy, "predict_field", check_controller)
    state = SimpleNamespace(current_observation={}, pose=[0., 0., 30.])
    first = obj.predict(state, obj.record["instruction"])
    second = obj.predict(state, obj.record["instruction"])
    assert seen == [2, 2, 2, 2]
    assert first.diagnostics["stable_navigation_goal"]["event"] == "initialized"
    assert second.diagnostics["stable_navigation_goal"]["held_goal_id"] == 2
    assert second.diagnostics["stable_navigation_goal"]["visual_top1_id"] == 3
    assert second.diagnostics["stable_navigation_goal"]["event"] == "pending"
