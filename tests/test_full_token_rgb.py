"""Full-token, per-candidate geometry and observed-RGB regression tests."""

import json
import numpy as np
import pytest
import torch

from graphnav.candidate_geometry import FullCandidateDataset, collate_full
from graphnav.full_localization import FullTokenGeometryLocalizer
from graphnav.landmark_alias import resolve_references, build_name_index
from graphnav.static_localization import cross_entropy_goal
from graphnav.full_experiment import _batch_inputs


def sample(tmp_path, *, with_landmark=True, with_rgb=False):
    rec = {
        "episode_id": "test_block_0:5:0:0",
        "map_name": "test_block_0",
        "dataset_version": "revised", "match_mode": "fuzzy",
        "instruction": "to the east of the BMTR Building near Riverside",
        "world_bounds": [0, 0, 400, 400], "physical_map_sizes": [400, 400],
        "target_xy": [0.55, 0.31],
        "landmarks": [
            {"id": "5", "name": "BMTR", "object_type": "Building",
             "aliases": ["BMTR Building"],
             "contour": [[.2, .25], [.3, .25], [.3, .35], [.2, .35]]},
            {"id": "7", "name": "Riverside", "object_type": "Footpath",
             "contour": [[.1, .4], [.8, .4], [.8, .45], [.1, .45]]},
        ] if with_landmark else [],
    }
    if with_rgb:
        rec["rgb_observations"] = [{"path": "observed.png",
                                    "footprint_world": [[0, 0], [10, 0], [10, 10], [0, 10]]}]
    filename = tmp_path / "one.jsonl"
    filename.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    dataset = FullCandidateDataset(
        filename, grid_size=4, max_landmarks=3, max_roads=0,
        max_relations=3, visual_required=with_rgb,
    )
    return collate_full([dataset[0], dataset[0]])


def test_full_candidate_geometry_one_polygon_one_node(tmp_path):
    batch = sample(tmp_path)
    assert batch["pair_geometry"].shape == (2, 16, 6, 10)
    assert batch["entity_valid"].sum(dim=1).tolist() == [2, 2]
    assert batch["node_valid"].sum() >= 4
    # Candidate at row1,col1 is east of BMTR center:
    east = 1 * 4 + 2
    west = 1 * 4 + 0
    assert batch["pair_geometry"][0, east, 0, 0] > 0
    assert batch["pair_geometry"][0, west, 0, 0] < 0
    assert batch["pair_geometry"][0, east, 0, 2] >= 0


def test_full_tokens_and_geometry_gat_backward(tmp_path):
    torch.manual_seed(123)
    torch.set_num_threads(2)
    batch = sample(tmp_path)
    network = FullTokenGeometryLocalizer(
        hidden=32, text_dim=12, heads=4, max_relations=3,
        vision_dim=6, top_k=5, dropout=0,
    )
    token_states = torch.randn(2, 7, 12)
    token_mask = torch.tensor([[1,1,1,1,0,0,0],[1,1,1,1,1,0,0]],dtype=torch.bool)
    names = torch.randn(2, 6, 12)
    for variant in ("building_prior", "prior", "rules", "tokens", "geometry", "gat"):
        result = network(batch, variant=variant, tokens=token_states,
                         token_mask=token_mask, node_emb=names)
        assert result.belief_logits.shape == (2, 16)
        assert result.candidate_xy.shape == (2, 5, 2)
        assert torch.isfinite(result.belief_logits).all()
    pred = network(batch, variant="gat", tokens=token_states,
                   token_mask=token_mask, node_emb=names)
    loss = cross_entropy_goal(pred.belief_logits, batch["target_xy"], grid_size=4)
    loss.backward()
    assert network.token_attn.in_proj_weight.grad is not None
    assert network.gnn.layer1.edge.weight.grad is not None
    assert network.geometry_proj[0].weight.grad is not None
    assert network.gnn.layer1.edge.weight.grad.abs().sum() > 0


def test_visual_refinement_does_not_use_unseen_pixels(tmp_path):
    batch = sample(tmp_path)
    model = FullTokenGeometryLocalizer(hidden=32, text_dim=12, heads=4,
                                      max_relations=3, vision_dim=6, top_k=4,
                                      dropout=0).eval()
    with torch.no_grad():
        model.visual_score[-1].weight.fill_(0.1)
        model.visual_score[-1].bias.fill_(0.03)
    token = torch.randn(2, 6, 12)
    mask = torch.ones(2, 6, dtype=torch.bool)
    names = torch.randn(2, 6, 12)
    source = model(batch,variant="gat",tokens=token,token_mask=mask,node_emb=names)
    zeros = torch.zeros(2,4)
    first = model.refine_rgb(source,torch.randn(2,4,6),zeros)
    second = model.refine_rgb(source,torch.randn(2,4,6)*100,zeros)
    assert torch.equal(source.belief_logits,first.belief_logits)
    assert torch.equal(first.belief_logits,second.belief_logits)
    one_visible = zeros.clone()
    one_visible[0,0] = 0.9
    third = model.refine_rgb(source,torch.randn(2,4,6),one_visible)
    assert not torch.equal(source.belief_logits[0],third.belief_logits[0])
    assert torch.equal(source.belief_logits[1],third.belief_logits[1])
    edited = ((third.belief_logits[0]-source.belief_logits[0]).abs() > 1e-7)
    assert edited.sum() == 1
    assert edited[source.candidate_ids[0,0]]


def test_empty_anchor_graph_still_finite_and_is_properly_masked(tmp_path):
    batch = sample(tmp_path,with_landmark=False)
    assert batch["entity_valid"].sum()==0
    model = FullTokenGeometryLocalizer(
        hidden=32,text_dim=12,heads=4,max_relations=3,vision_dim=6,
        top_k=4,dropout=0,
    )
    output = model(batch,variant="gat",
                   tokens=torch.randn(2,4,12),
                   token_mask=torch.ones(2,4,dtype=torch.bool),
                   node_emb=torch.zeros(2,6,12))
    assert torch.isfinite(output.belief_logits).all()


def test_fuzzy_is_opt_in_with_type_safety_and_ambiguity_abstention():
    objects = {
        "1":{"name":"Parklanne","object_type":"TrafficRoad",
             "contour":[[0,0],[1,0],[0,1]]},
        "2":{"name":"Parklanne","object_type":"Building",
             "contour":[[2,2],[3,2],[2,3]]},
        "3":{"name":"Parklance","object_type":"TrafficRoad",
             "contour":[[4,4],[5,4],[4,5]]},
    }
    idx=build_name_index(objects)
    assert not resolve_references(["Parklanne Road"],objects,index=idx,match_mode="exact")
    typed=resolve_references(["Parklanne Road"],objects,index=idx,match_mode="typed")
    assert [x["id"] for x in typed]==["1"]
    assert not resolve_references(["Parklaane Road"],objects,index=idx,match_mode="typed")
    fuzzy=resolve_references(["Parklaane Road"],objects,index=idx,
                             match_mode="fuzzy",fuzzy_threshold=.80,
                             fuzzy_margin=.01)
    assert all(x["object_type"]=="TrafficRoad" for x in fuzzy)


def test_frozen_batch_has_no_gt_forward_fields(tmp_path):
    batch=sample(tmp_path)
    class FakeBERT:
        def encode_tokens(self,strings):
            return torch.zeros(len(strings),5,12),torch.ones(len(strings),5,dtype=torch.bool)
        def encode(self,strings):
            return torch.zeros(len(strings),12)
    observable, states,mask,names=_batch_inputs(batch,FakeBERT(),torch.device("cpu"),"gat")
    assert "target_xy" not in observable and "rgb_observations" not in observable
    assert states.shape==(2,5,12)
    assert names.shape==(2,6,12)
