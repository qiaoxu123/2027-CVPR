import json
import numpy as np
import torch

from graphnav.data import (
    CityNavJSONL, bind_reference_landmarks, collate_episodes, resample_teacher_path
)
from graphnav.config import GraphNavConfig


def test_binding_is_exact_and_keeps_ambiguous_names():
    objects = {
        "1": {"name": "St Marys Church", "contour": [[0, 0], [1, 0], [1, 1]]},
        "2": {"name": "St Marys Church", "contour": [[2, 2], [3, 2], [2, 3]]},
        "3": {"name": "Old School", "contour": [[3, 3], [4, 3], [3, 4]]},
    }
    matched = bind_reference_landmarks(["st marys church"], objects, 16)
    assert [lm["id"] for lm in matched] == ["1", "2"]
    assert bind_reference_landmarks(["hospital"], objects, 16) == []


def test_teacher_path_is_not_replaced_with_goal_shortcut():
    path = [[0, 0], [0, 1], [1, 1]]
    sampled = resample_teacher_path(path, future_start=0, steps=3)
    assert np.allclose(sampled, path)
    assert not np.allclose(sampled, [[0, 0], [0.5, 0.5], [1, 1]])


def test_loader_does_not_include_ground_truth_in_model_arguments(tmp_path):
    cfg = GraphNavConfig(grid_size=4, max_landmarks=2, contour_points=8,
                         trajectory_waypoints=3)
    entry = {
        "instruction": "east of church",
        "landmarks": [{"name": "church", "contour": [[.1, .1], [.3, .1],
                                                      [.3, .3], [.1, .3]]}],
        "uav_pose": [.1, .2, 0, 1],
        "target_xy": [.5, .2],
        "teacher_future": [[.2, .2], [.3, .2], [.4, .2]],
        "physical_map_sizes": [400, 400],
    }
    path = tmp_path / "citynav.jsonl"
    path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    ds = CityNavJSONL(path, cfg)
    batch = collate_episodes([ds[0], ds[0]])
    assert batch["landmark_contours"].shape == (2, 2, 8, 2)
    assert batch["target_xy"].shape == (2, 2)
    assert batch["landmark_valid"].sum() == 2
