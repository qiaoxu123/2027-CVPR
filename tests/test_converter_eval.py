import json
import numpy as np
import torch

from graphnav.config import GraphNavConfig
from graphnav.model import GraphNav
from graphnav.evaluate import open_loop_metrics
from graphnav.data import CityNavJSONL
from scripts.prepare_citynav import convert


def test_converter_preserves_name_contour_and_causal_pose_prefix(tmp_path):
    obj = {
        "city_block_0": {
            "7": {"name": "", "descriptions": ["east of the Church"],
                  "contour": [[230, 200], [231, 200], [230, 201]]},
            "9": {"name": "Church", "descriptions": [],
                  "contour": [[120, 120], [130, 120], [130, 130], [120, 130]]},
        }
    }
    proc = {"city_block_0": {"7": [{"landmarks": ["Church"]}]}}
    trajectory = [{
        "area": "city", "block": "0", "object_ids": [7], "ann_ids": [0],
        "target_positions": [[230, 210, 60]],
        "trajectory": [[100, 100, 60, 0.0, 0.0],
                       [120, 100, 60, 0.0, 0.0],
                       [140, 100, 60, 0.0, 0.0]],
    }]
    out = tmp_path / "converted.jsonl"
    count, skipped = convert(
        obj, proc, trajectory, {"city_block_0": [0, 0, 400, 400]},
        out, grid=4, stride=1, waypoints=3, max_landmarks=2,
        ground_levels={"city_block_0": 0}, max_steps=3,
    )
    assert (count, skipped) == (3, 0)
    records = [json.loads(x) for x in out.read_text().splitlines()]
    assert records[0]["landmarks"][0]["name"] == "Church"
    assert records[0]["rgb_observations"] == []
    assert len(records[0]["teacher_future"]) == 3
    assert records[0]["target_xy"] == records[2]["target_xy"]
    assert records[0]["uav_pose"] != records[1]["uav_pose"]
    ds = CityNavJSONL(out, GraphNavConfig(
        grid_size=4, max_landmarks=2, contour_points=8, trajectory_waypoints=3
    ))
    assert len(ds) == 3


def test_open_loop_recalls_have_consistent_denominators():
    cfg = GraphNavConfig(text_dim=8, hidden_dim=16, heads=4,
                         grid_size=4, max_landmarks=2, contour_points=8,
                         goal_top_k=3)
    model = GraphNav(cfg)
    # Avoid dependence on pretrained text encoder for metrics tests.
    logits = torch.zeros((1, 16))
    logits[0, 5] = 10
    result = type("Dummy", (), {"belief_logits": logits})()
    target = torch.tensor([[0.375, 0.375]])
    stats = open_loop_metrics(result, target, torch.tensor([[400.0, 400.0]]), 4)
    assert stats["samples"] == 1
    assert stats["R@1/20m_hits"] == 1
    assert stats["R@5/20m_hits"] >= stats["R@1/20m_hits"]


def test_static_conversion_does_not_relabel_roads_as_buildings_or_need_trajectory(tmp_path):
    obj = {'city_block_0': {
        '7': {'name': '', 'descriptions': ['near Church along Main Road']},
        '8': {'name': 'Main Road', 'object_type': 'TrafficRoad',
              'contour': [[0, 0], [100, 0], [100, 20]]},
        '9': {'name': 'Church', 'object_type': 'Building',
              'contour': [[120, 120], [130, 120], [130, 130]]}}}
    proc = {'city_block_0': {'7': [{'landmarks': ['Main Road', 'Church']}]}}
    episodes = [{'area': 'city', 'block': 0, 'object_ids': [7], 'ann_ids': [0],
                 'target_positions': [[200, 210, 60]]}]
    filename = tmp_path / 'static.jsonl'
    assert convert(obj, proc, episodes, {'city_block_0': [0, 0, 400, 400]},
                   filename, static_only=True) == (1, 0)
    record = json.loads(filename.read_text())
    assert [r['id'] for r in record['landmarks']] == ['8', '9']
    assert [r['object_type'] for r in record['landmarks']] == ['TrafficRoad', 'Building']
    assert all('contour' in r and 'polyline' not in r for r in record['landmarks'])
    assert not {'uav_pose', 'teacher_future', 'rgb_observations'} & record.keys()
