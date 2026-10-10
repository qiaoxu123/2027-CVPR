"""GT-free Stop guards and joint appearance control regressions."""
import inspect

import numpy as np
import pytest
import torch
from torch import nn

from graphnav.full_localization import GoalField
from graphnav.joint_active import JointActiveGraphNav
from graphnav.joint_inference_controls import joint_stop_decision


def decision(goal=(60., 50.), *, probability=.9, guard=20., pose=(50., 50., 40., 0.),
             threshold=.5):
    return joint_stop_decision(
        pose_world=pose, candidate_world_xy=goal,
        stop_probability=probability, ground_level=20.,
        threshold=threshold, guard_radius_m=guard,
    )


def test_stop_requires_probability_predicted_goal_and_legal_altitude():
    stop, diag = decision()
    assert stop
    assert diag["stop_predicted_goal_distance_m"] == pytest.approx(10.)
    assert diag["stop_reject_reason"] == "none"
    stop, diag = decision(goal=(200., 50.))
    assert not stop and diag["stop_reject_reason"] == "predicted_goal_distance"
    stop, diag = decision(probability=.2)
    assert not stop and diag["stop_reject_reason"] == "probability"
    stop, diag = decision(pose=(50., 50., 140., 0.))
    assert not stop and diag["stop_reject_reason"] == "altitude"


def test_stop_guard_zero_is_explicit_unprotected_ablation():
    assert decision(goal=(200., 50.), guard=0.)[0]
    assert not decision(goal=(200., 50.), guard=20.)[0]
    assert not decision(goal=(60., 50.), probability=.3, guard=0.)[0]


def test_stop_never_reads_gt_or_future_and_rejects_invalid_params():
    params = set(inspect.signature(joint_stop_decision).parameters)
    assert not {"target", "target_world", "teacher_world", "future", "success"} & params
    for invalid in (
        dict(pose_world=(50., 50., float("nan"), 0.)),
        dict(guard_radius_m=-2.),
        dict(stop_probability=1.5),
        dict(threshold=1.2),
    ):
        data = dict(pose_world=(50., 50., 40., 0.), candidate_world_xy=(60., 50.),
                    stop_probability=.9, ground_level=20.)
        data.update(invalid)
        with pytest.raises(ValueError):
            joint_stop_decision(**data)


class FrozenToyA3(nn.Module):
    hidden = 8
    top_k = 3

    def forward(self, batch, **kwargs):
        grid = batch["grid_xy"]
        logits = torch.tensor([[.4, .3, .2, .1]], device=grid.device)
        context = torch.zeros(1, 4, 8, device=grid.device)
        ids = logits.topk(3, dim=-1).indices
        return GoalField(
            logits, ids,
            grid.gather(1, ids[..., None].expand(-1, -1, 2)),
            context.gather(1, ids[..., None].expand(-1, -1, 8)),
            dense_context=context,
        )


def test_no_belief_causally_removes_visual_logits_but_keeps_coverage():
    model = JointActiveGraphNav(FrozenToyA3(), vision_dim=8, matcher_hidden=4).eval()
    with torch.no_grad():
        model.visual_gain.fill_(2.)
    images = torch.zeros(1, 4, 3, 8)
    images[0, 0, 0, 0] = 1.
    images[0, 1, 0, 0] = -1.
    valid = torch.zeros(1, 4, 3, dtype=torch.bool)
    valid[0, :2, 0] = True
    inputs = dict(
        static_inputs={"grid_xy": torch.tensor([[
            [.2, .2], [.4, .4], [.6, .6], [.8, .8],
        ]])},
        tokens=torch.zeros(1, 1, 8),
        token_mask=torch.ones(1, 1, dtype=torch.bool),
        node_emb=torch.zeros(1, 1, 8),
        native_text=torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0.]]),
        roi_features=images, roi_valid=valid, roi_quality=torch.zeros(1, 4, 3, 3),
        pose_world=torch.tensor([[50., 50., 40., 0.]]),
        bounds=torch.tensor([[0., 0., 100., 100.]]),
        ground_level=torch.tensor([20.]),
    )
    original = model(**inputs)
    no_visual = model(**inputs, visual_control="no-belief")
    torch.testing.assert_close(no_visual.belief.belief_logits,
                               no_visual.static.belief_logits, rtol=0, atol=0)
    assert torch.equal(no_visual.observed, original.observed)
    assert torch.any(original.belief.belief_logits != no_visual.belief.belief_logits)
    with pytest.raises(ValueError, match="visual-belief"):
        model(**inputs, visual_control="fake-control")
