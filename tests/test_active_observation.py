"""Active XY/Z observation planning and current-frame multi-scale crop tests."""
from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest

from graphnav.active_observation import (
    ActiveObservationConfig, appearance_size_proxy_m, estimated_object_pixels,
    plan_active_observation,
)
from graphnav.observed_scales import extract_observed_scale, multiscale_observed_crops


def observation(*, support=True):
    rgb = np.full((384, 384, 3), 120, dtype=np.uint8)
    valid = np.full((384, 384), support, dtype=bool)
    return dict(
        bgr=rgb, depth_valid=valid,
        footprint_world=np.array([[0, 0], [100, 0], [100, 100], [0, 100]], np.float32),
    )


def plan(pose, goals, scores, coverage=None, **extras):
    goals = np.asarray(goals, dtype=float)
    return plan_active_observation(
        pose_xyzyaw=pose, map_bounds=[0, 0, 200, 200],
        ground_level=0., candidate_xy=goals / 200,
        candidate_logits=scores,
        observed_confidence=np.zeros(len(goals)) if coverage is None else coverage,
        instruction="the white car near a brown house",
        step=0, **extras,
    )


def test_high_uncertainty_candidate_is_approached_only_one_bounded_step():
    decision = plan(
        [0, 0, 80, 0], [[150, 150], [70, 70]], [3, 2],
    )
    assert decision.action == "approach"
    assert np.linalg.norm(decision.xyz[:2]) <= 50 + 1e-6
    assert decision.xyz[2] == 80


def test_bounded_descent_can_improve_ideal_pixel_count():
    before = estimated_object_pixels(4., 80.)
    decision = plan([100, 100, 80, 0],
                    [[100, 100], [104, 104]], [3, 2])
    assert decision.action == "descend"
    assert decision.xyz[2] == 70
    assert estimated_object_pixels(4., 70.) > before


def test_lateral_is_xy_not_fake_perspective_camera():
    decision = plan(
        [100, 100, 60, 0], [[100, 100], [102, 102]], [3, 2],
        config=ActiveObservationConfig(enable_descend=False),
    )
    assert decision.action == "lateral"
    assert decision.xyz[2] == 60
    assert np.linalg.norm(decision.xyz[:2] - np.array([100, 100])) <= 12 + 1e-6
    assert decision.xyz.shape == (3,)
    assert "yaw" not in {f.name for f in fields(type(decision))}


def test_action_budget_and_invalid_source_altitude_are_safe_fallback():
    assert plan(
        [100, 100, 80, 0], [[100, 100], [110, 110]], [1, 0],
        used_actions=2,
    ) is None
    assert plan(
        [100, 100, 80, 0], [[100, 100], [110, 110]], [1, 0],
        step=18,
    ) is None
    assert plan(
        [100, 100, 140, 0], [[100, 100], [110, 110]], [1, 0],
    ) is None
    assert plan(
        [100, 100, 80, 0], [[100, 100], [110, 110]], [3, 2],
        coverage=[1., 1.],
    ) is None


def test_invalid_observation_confidence_is_rejected():
    with pytest.raises(ValueError, match="confidence"):
        plan([100, 100, 80, 0], [[100, 100]], [1], coverage=[2])


def test_crop_is_current_frame_only_not_superresolution():
    current = observation()
    crop = extract_observed_scale(current, [50., 50.], extent_m=40.)
    assert crop is not None
    assert crop.bgr.shape == (384, 384, 3)
    assert crop.native_span_px < 384.
    assert crop.native_span_px > 100
    assert crop.valid_fraction == 1.
    assert set(multiscale_observed_crops(current, [50, 50], extents_m=[32, 64])) == {32., 64.}


def test_unobserved_or_clipped_crop_cannot_be_used():
    assert extract_observed_scale(observation(), [99., 99.], extent_m=32.) is None
    assert extract_observed_scale(observation(support=False), [50., 50.], extent_m=40.) is None
    with pytest.raises(ValueError, match="ROI scale"):
        extract_observed_scale(observation(), [50, 50], extent_m=-1)


def test_small_object_proxy_is_only_heuristic():
    assert appearance_size_proxy_m("a red car by the church") == 4.
    assert appearance_size_proxy_m("a brown house") == 15.
    with pytest.raises(ValueError):
        estimated_object_pixels(4., 0)
