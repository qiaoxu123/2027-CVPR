"""TRAIN-only explicit hierarchy: masks, invariance, label isolation and gradients."""
import json

import pytest
import torch

from graphnav.hierarchical_relation_graph import (
    HierarchicalRelationGraph, RegionAugmentedCandidateDataset, relation_mode,
)


def _inputs():
    torch.manual_seed(42)
    b, q, a, r, td, cd = 3, 16, 3, 2, 12, 8
    return dict(
        base_logits=torch.randn(b, q),
        candidate_context=torch.randn(b, q, cd),
        tokens=torch.randn(b, 5, td),
        token_mask=torch.ones(b, 5, dtype=torch.bool),
        anchor_embeddings=torch.randn(b, a, td),
        anchor_valid=torch.tensor([
            [True, True, False], [True, False, False], [False, False, False]
        ]),
        pair_geometry=torch.randn(b, q, a, 10),
        region_grid_geometry=torch.randn(b, q, r, 10),
        region_anchor_geometry=torch.randn(b, r, a, 10),
        region_valid=torch.tensor([
            [True, True], [False, False], [True, False]
        ]),
        region_types=torch.tensor([[1, 1], [0, 0], [1, 0]]),
        relation_mode=torch.tensor([2, 2, 1]),
    )


def _model():
    return HierarchicalRelationGraph(context_dim=8, text_dim=12, hidden=16).eval()


def test_parser_conservative_and_indirect():
    assert relation_mode("A red car in the parking lot next to One Stop") == 2
    assert relation_mode("The red car near One Stop") == 1
    assert relation_mode("The white building behind the cafe") == 0
    assert relation_mode("A triangular parking lot next to One Stop") == 1
    assert relation_mode("A building inside Trinity College") == 3


def test_direct_inside_keeps_node_probability_normalized():
    inp = _inputs()
    inp["relation_mode"][0] = 3
    out = _model()(**inp)
    assert out["direct_active"][0]
    assert not out["indirect_active"][0]
    torch.testing.assert_close(out["probabilities"].sum(1), torch.ones(3))


def test_missing_intermediary_or_anchors_must_abstain():
    inp = _inputs()
    out = _model()(**inp)
    assert out["indirect_active"].tolist() == [True, False, False]
    assert out["unresolved"].tolist() == [False, True, True]
    torch.testing.assert_close(out["logits"][1:], inp["base_logits"][1:], atol=0, rtol=0)
    torch.testing.assert_close(
        out["probabilities"].sum(-1), torch.ones(3), atol=1e-6, rtol=1e-6
    )


def test_anchor_and_intermediate_order_invariant():
    model = _model()
    inp = _inputs()
    before = model(**inp)["logits"]
    anchor_perm = torch.tensor([2, 0, 1])
    region_perm = torch.tensor([1, 0])
    reordered = dict(inp)
    reordered["anchor_embeddings"] = inp["anchor_embeddings"][:, anchor_perm]
    reordered["anchor_valid"] = inp["anchor_valid"][:, anchor_perm]
    reordered["pair_geometry"] = inp["pair_geometry"][:, :, anchor_perm]
    reordered["region_anchor_geometry"] = inp["region_anchor_geometry"][
        :, region_perm
    ][:, :, anchor_perm]
    reordered["region_grid_geometry"] = inp["region_grid_geometry"][
        :, :, region_perm
    ]
    reordered["region_valid"] = inp["region_valid"][:, region_perm]
    reordered["region_types"] = inp["region_types"][:, region_perm]
    after = model(**reordered)["logits"]
    torch.testing.assert_close(before, after, atol=1e-5, rtol=1e-5)


def test_gradient_and_flat_ablation():
    model = _model().train()
    inp = _inputs()
    out = model(**inp)
    loss = out["logits"][0].square().sum()
    loss.backward()
    assert model.head[-1].weight.grad is not None
    assert torch.isfinite(model.head[-1].weight.grad).all()
    assert model.head[-1].weight.grad.abs().sum() > 0
    flat = model(**inp, ablation="flat")
    assert flat["direct_active"].tolist() == [True, True, False]
    assert not flat["indirect_active"].any()


def test_invalid_shapes_rejected():
    inp = _inputs()
    inp["pair_geometry"] = inp["pair_geometry"][:, :-1]
    with pytest.raises(ValueError, match="invalid candidate"):
        _model()(**inp)


def test_public_region_builder_does_not_read_target_label(tmp_path):
    rec = {
        "dataset_version": "revised",
        "match_mode": "fuzzy",
        "episode_id": "cambridge_block_0:42:0:0",
        "map_name": "cambridge_block_0",
        "world_bounds": [0, 0, 200, 200],
        "physical_map_sizes": [200, 200],
        "instruction": "The red car in the parking lot next to One Stop",
        "landmarks": [{
            "id": "7", "name": "One Stop", "object_type": "Building",
            "contour": [[0.1, 0.1], [0.2, 0.1], [0.2, 0.2], [0.1, 0.2]]
        }],
        "target_xy": [0.4, 0.4],
    }
    other = dict(rec, episode_id="cambridge_block_0:43:0:0", target_xy=[0.8, 0.8])
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps(rec) + "\n" + json.dumps(other) + "\n", encoding="utf-8"
    )
    objs = {
        "cambridge_block_0": {
            "8": {
                "object_type": "Parking",
                "contour": [[60, 60], [90, 60], [90, 90], [60, 90]]
            },
            "9": {
                "object_type": "Parking",
                "contour": [[150, 150], [170, 150], [170, 170], [150, 170]]
            }
        }
    }
    objects_path = tmp_path / "objects.json"
    objects_path.write_text(json.dumps(objs), encoding="utf-8")
    ds = RegionAugmentedCandidateDataset(
        path, objects_json=objects_path, max_regions=2,
        max_landmarks=4, max_roads=0, max_relations=2,
        deduplicate_queries=False,
    )
    a, b = ds[0], ds[1]
    for key in (
        "pair_geometry", "region_grid_geometry",
        "region_anchor_geometry", "region_valid", "region_types",
    ):
        torch.testing.assert_close(a[key], b[key])
    assert a["relation_mode"] == b["relation_mode"] == 2
    assert a["region_valid"].tolist() == [True, True]
    assert not torch.equal(a["target_xy"], b["target_xy"])
