"""Landmark acquisition is vision-evidence driven, not just Polygon overlap."""
import torch
from torch import nn

from graphnav.landmark_acquisition import (
    AcquisitionThresholds, LandmarkAcquisition,
)
from graphnav.unified_pgvb import UnifiedPGVB
from graphnav.full_localization import GoalField


class TinyGeometry(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.1))
        self.calls = 0

    def forward(self, batch, *, variant, tokens, token_mask, node_emb):
        assert variant == 'geometry'
        self.calls += 1
        grid = batch['grid_xy']
        logits = torch.linspace(0, 1, grid.shape[1])[None] + self.bias
        dense = grid[..., :1].expand(-1, -1, 128).contiguous()
        ids = logits.topk(20, dim=1).indices
        return GoalField(
            belief_logits=logits, candidate_ids=ids,
            candidate_xy=grid.gather(1, ids[..., None].expand(-1, -1, 2)),
            candidate_context=dense.gather(1, ids[..., None].expand(-1, -1, 128)),
            dense_context=dense,
        )


def _inputs():
    v = (torch.arange(28) + .5) / 28
    ys, xs = torch.meshgrid(v, v, indexing='ij')
    return dict(
        static_inputs={'grid_xy': torch.stack((xs.flatten(), ys.flatten()), -1)[None]},
        tokens=None, token_mask=None, node_emb=None,
    )


def _patches():
    scores = torch.tensor([[5., 4., 4., 5., 0., 1., 0., 1.]])
    counterfactual = scores.roll(shifts=4, dims=1)
    labels = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.int8)
    valid = torch.ones_like(labels, dtype=torch.bool)
    return scores, counterfactual, labels, valid


def test_polygon_visibility_alone_never_acquires():
    tracker = LandmarkAcquisition()
    for _ in range(5):
        audit = tracker.observe()
        assert audit["visual_gain"] == 0
        assert tracker.state == "geometry"
        assert not audit["identity_independently_verified"]


def test_two_separate_rgb_confirmations_before_switch():
    tracker = LandmarkAcquisition()
    args = _patches()
    first = tracker.observe(*args)
    assert first["state"] == "candidate"
    assert first["visual_gain"] == 0
    second = tracker.observe(*args)
    assert second["state"] == "tracking"
    assert second["visual_gain"] == tracker.thresholds.max_visual_gain / 2
    assert second["visual_auc"] == 1.
    assert second["counterfactual_auc"] == 0.
    third = tracker.observe(*args)
    assert third["state"] == "tracking"
    assert third["visual_gain"] == tracker.thresholds.max_visual_gain
    assert third["acquisition_events"] == 1


def test_geometric_labels_with_non_discriminative_rgb_are_not_confirmation():
    tracker = LandmarkAcquisition()
    scores, _, labels, valid = _patches()
    for _ in range(4):
        # Same ROI and identical scores for true / shuffled features.
        audit = tracker.observe(scores, scores.clone(), labels, valid)
    assert audit["state"] == "geometry"
    assert audit["reason"] == "weak_rgb_discrimination"
    assert audit["visual_gain"] == 0.


def test_incomplete_negatives_and_unknown_pixels_cannot_trigger():
    tracker = LandmarkAcquisition()
    scores, fake, labels, valid = _patches()
    labels[:] = -1
    labels[0, 0] = 1
    for _ in range(4):
        audit = tracker.observe(scores, fake, labels, valid)
    assert audit["state"] == "geometry"
    assert audit["positive_patches"] == 1
    assert audit["visual_gain"] == 0


def test_lost_frame_coasts_without_stale_visual_residual():
    tracker = LandmarkAcquisition()
    for _ in range(2):
        tracker.observe(*_patches())
    lost = tracker.observe()
    assert lost["state"] == "coasting"
    assert lost["visual_gain"] == 0
    tracker.observe()
    gone = tracker.observe()
    assert gone["state"] == "geometry"
    assert gone["visual_gain"] == 0
    tracker.reset()
    assert tracker.acquisitions == 0
    assert tracker.confirmations == 0


class ToyVisual(nn.Module):
    def forward(self, *, image_features, name_text, type_text,
                polygon_tokens, valid):
        return image_features[..., 0, 0].float().masked_fill(~valid, 0.)


def test_integrated_goalfield_switch_is_rgb_triggered_and_uses_one_visual_head():
    model = UnifiedPGVB(geometry=TinyGeometry()).eval().requires_grad_(False)
    model.pgvb = ToyVisual()
    tracker = LandmarkAcquisition()
    scores, _, labels, valid = _patches()
    features = torch.zeros(1, 8, 3, 1024)
    features[:, :, 0, 0] = scores
    xy = torch.tensor([[[
        .10, .10], [.15, .10], [.10, .15], [.15, .15],
        [.70, .70], [.75, .70], [.70, .75], [.75, .75],
    ]])
    observed = dict(
        local_uv=xy, image_features=features,
        name_text=torch.zeros(1, 1024),
        type_text=torch.zeros(1, 1024),
        polygon_tokens=torch.zeros(1, 16, 4), valid=valid,
    )
    initial = model.forward_field(**_inputs(), observed=observed,
                                  acquisition=tracker, anchor_labels=labels)
    assert initial["field"] is initial["geometry_field"]
    assert initial["audit"]["acquisition"]["state"] == "candidate"
    confirmed = model.forward_field(**_inputs(), observed=observed,
                                    acquisition=tracker, anchor_labels=labels)
    assert confirmed["audit"]["acquisition"]["state"] == "tracking"
    assert confirmed["audit"]["acquisition"]["visual_gain"] > 0
    assert model.geometry.calls == 1
    assert confirmed["field"].candidate_context.shape == (1, 20, 128)
    model.reset_episode()
    tracker.reset()
    fresh = model.forward_field(**_inputs(), observed=None,
                                acquisition=tracker)
    assert fresh["field"] is fresh["geometry_field"]
    assert tracker.state == "geometry"


def test_tracking_does_not_use_distance_or_gt():
    tracker = LandmarkAcquisition()
    audit = tracker.observe(*_patches())
    assert not any("distance" in key or "ground_truth" in key for key in audit)
    assert audit["identity_independently_verified"] is False


def test_nearby_public_geometry_is_reported_only_after_arrival(monkeypatch):
    from types import SimpleNamespace
    from shapely.geometry import box
    from graphnav.a3_runtime import A3TrajectoryPolicy
    from graphnav.unified_pgvb_runtime import UnifiedPGVBNamedLandmarkPolicy

    policy = object.__new__(UnifiedPGVBNamedLandmarkPolicy)
    policy.record = {"instruction": "Navigate to Alpha Hall."}
    policy.mode = "geometry"
    policy._initialize_geometry_inputs = lambda: None
    policy._observed_inputs = lambda obs: (None, "geometry_only")
    policy.inputs, policy.tokens, policy.mask, policy.names = {}, None, None, None
    policy.device = torch.device("cpu")
    policy.bound = .5
    policy.pgvb_calls, policy.pgvb_seconds = 0, 0.
    policy.audit_callback = None
    policy.instruction_spec = {
        "version": "unified-instruction-v1", "intent": "navigate_to_landmark",
        "landmark": {"public_ids": ["alpha"]},
    }
    policy._public_region = box(0, 0, 10, 10)
    policy._neighbor_geometry = [{"public_id": "parking", "type": "Parking"}]
    policy._last_observation_metadata = None
    policy._landmark = None
    policy.unified = type("FakeModel", (), {
        "direction_ready": False,
        "forward_structured": lambda self, spec, **kwargs: {
            "geometry_field": None, "field": None, "evidence": None,
            "audit": {"active": False, "covered_cells": 0,
                      "max_abs_delta": 0., "changed_top1": False},
        },
    })()
    monkeypatch.setattr(
        A3TrajectoryPolicy, "predict_field",
        lambda *args: type("Prediction", (), {"diagnostics": {}})(),
    )
    far = policy.predict(
        SimpleNamespace(current_observation={}, pose=[100., 100., 30.]),
        policy.record["instruction"],
    )
    assert not far.diagnostics["near_landmark_geometry"]["within_20m"]
    assert far.diagnostics["near_landmark_geometry"]["nearby_object_count"] == 0
    near = policy.predict(
        SimpleNamespace(current_observation={}, pose=[5., 5., 30.]),
        policy.record["instruction"],
    )
    assert near.diagnostics["near_landmark_geometry"]["within_20m"]
    assert near.diagnostics["near_landmark_geometry"]["nearby_object_count"] == 1
    assert near.diagnostics["landmark_visual_gain"] == 0
