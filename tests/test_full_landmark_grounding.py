"""Regression: typed CityRefer surface grounding without any visual/GT leakage."""

import json
import numpy as np
import torch

from graphnav.data import bind_reference_landmarks
from graphnav.landmark_alias import build_name_index, type_aware_alias_match
from graphnav.semantic_geometry import (
    EDGE_TYPES, NODE_TYPES, build_entities, parse_spatial_relations,
    prior_and_rule_fields,
)
from graphnav.semantic_graph import prepare_semantic_graph, TypedGeometryGAT
from graphnav.static_localization import StaticLocalizationDataset, collate_static


def square(x, y, half=0.01):
    return [[x-half, y-half], [x+half, y-half],
            [x+half, y+half], [x-half, y+half]]


def fixtures():
    return {
        "a": {"name": "Broadway", "object_type": "TrafficRoad",
              "contour": square(.4, .4, .08)},
        "b": {"name": "BMTR", "object_type": "Building",
              "contour": square(.2, .2)},
        "c": {"name": "BMTR", "object_type": "Building",
              "contour": square(.7, .2)},
        "d": {"name": "Riverside", "object_type": "Footpath",
              "contour": square(.1, .8)},
        "e": {"name": "Riverside", "object_type": "Building",
              "contour": square(.7, .8)},
        "f": {"name": "Broadway", "object_type": "Building",
              "contour": square(.8, .8)},
    }


def test_exact_name_preferred_to_typed_alias_and_homonyms_kept():
    objects = fixtures()
    index = build_name_index(objects)
    matches = bind_reference_landmarks(
        ["BMTR building", "Broadway Road"], objects, 16, index=index
    )
    assert [m["id"] for m in matches] == ["b", "c", "a"]
    assert matches[0]["aliases"] == ["BMTR building"]
    assert matches[2]["object_type"] == "TrafficRoad"
    assert type_aware_alias_match("One Stop building", "One Stop", "Building")
    assert not type_aware_alias_match("One Stop building", "One Stop", "TrafficRoad")
    assert not type_aware_alias_match("One Stop street", "One Stop", "Building")
    exact = bind_reference_landmarks(["Broadway"], objects, 16, index=index)
    assert {m["id"] for m in exact} == {"a", "f"}


def test_traffic_road_surface_is_polygon_not_centerline_or_topology():
    objects = fixtures()
    selection = bind_reference_landmarks(
        ["Broadway Road", "Riverside footpath"], objects, 16
    )
    normalized = [
        {**item, "contour": item["contour"]} for item in selection
    ]
    graph = prepare_semantic_graph(
        normalized, [], "along Broadway Road near Riverside footpath",
        [0, 0, 400, 400], max_landmarks=4, max_roads=0,
        max_relations=4,
    )
    assert len(graph["entities"]) == 2
    assert graph["entities"][0]["kind"] == "road_surface"
    assert graph["entities"][1]["kind"] == "footpath"
    assert graph["node_types"][0] == NODE_TYPES["road_surface"]
    assert graph["node_types"][1] == NODE_TYPES["footpath"]
    assert graph["relations"][0].kind == "near" or graph["relations"][0].kind == "along"
    assert {r.kind for r in graph["relations"]} == {"along", "near"}
    assert not np.any(graph["edges"][:, :, 8 + EDGE_TYPES["roads_connected"]])
    assert all(len(x["geometry_m"]) == 4 for x in graph["entities"])
    center = np.asarray([[.4, .4]], np.float32) * 400.
    outside = np.asarray([[.95, .95]], np.float32) * 400.
    near, _, _ = prior_and_rule_fields(center, graph["entities"], graph["relations"])
    far, _, _ = prior_and_rule_fields(outside, graph["entities"], graph["relations"])
    assert near[0] > far[0]


def test_grounded_typed_alias_has_relation_node_and_gat_gradients():
    objs = fixtures()
    lm = bind_reference_landmarks(["Broadway Road", "BMTR building"], objs, 16)
    graph = prepare_semantic_graph(
        lm, [], "to the east of BMTR building and along Broadway Road",
        [0, 0, 400, 400], max_landmarks=4, max_roads=0, max_relations=4
    )
    assert {r.kind for r in graph["relations"]} == {"east", "along"}
    assert {x["kind"] for x in graph["entities"]} == {"building", "road_surface"}
    assert [x["id"] for x in graph["entities"]] == ["a", "b", "c"]
    assert graph["node_mask"].sum() == 5
    net = TypedGeometryGAT(hidden=16, text_dim=8, heads=4, dropout=0)
    kwargs = dict(
        nodes=torch.from_numpy(graph["nodes"]).unsqueeze(0),
        names=torch.randn(1, len(graph["nodes"]), 8),
        types=torch.from_numpy(graph["node_types"]).unsqueeze(0),
        adj=torch.from_numpy(graph["adjacency"]).unsqueeze(0),
        edges=torch.from_numpy(graph["edges"]).unsqueeze(0),
        valid=torch.from_numpy(graph["node_mask"]).unsqueeze(0),
    )
    h = net(**kwargs)
    assert h.shape == (1, len(graph["nodes"]), 16)
    h.square().sum().backward()
    assert net.layer1.edge.weight.grad is not None


def test_surface_static_dataset_invariant_to_target_pose_and_rgb(tmp_path):
    obj = fixtures()
    refs = bind_reference_landmarks(["Broadway Road", "BMTR building"], obj, 16)
    rec = {
        "episode_id": "map:target:0:0", "map_name": "map",
        "world_bounds": [0., 0., 400., 400.],
        "physical_map_sizes": [400., 400.],
        "instruction": "along Broadway Road near BMTR building",
        "landmarks": refs, "target_xy": [.8, .8],
    }
    file = tmp_path / "static.jsonl"
    file.write_text(json.dumps(rec) + "\n")
    ds = StaticLocalizationDataset(
        file, grid_size=8, max_landmarks=5, max_roads=0, max_relations=4,
    )
    x = ds[0]
    assert x["nonbuilding_surface_count"] == 1
    assert x["surface_count"] == 3
    assert x["prior"].shape == (64,)
    changed = {
        **rec, "target_xy": [.01, .02], "uav_pose": [.4, .2, 0., 1.],
        "rgb_observations": [{"path": "/data/private/future.png"}],
    }
    file.write_text(json.dumps(changed) + "\n")
    y = StaticLocalizationDataset(
        file, grid_size=8, max_landmarks=5, max_roads=0, max_relations=4
    )[0]
    for field in ("nodes", "node_types", "adjacency", "edges", "prior", "rules", "relation_fields"):
        assert torch.equal(x[field], y[field]), field


def test_only_matching_region_type_can_generate_prefix_alias():
    objects = {
        "1": {"name": "Main", "object_type": "Building", "contour": square(.3, .3)},
        "2": {"name": "Main", "object_type": "TrafficRoad", "contour": square(.5, .5)},
    }
    assert [m["id"] for m in bind_reference_landmarks(["Main Road"], objects, 16)] == ["2"]
    assert [m["id"] for m in bind_reference_landmarks(["Main building"], objects, 16)] == ["1"]
