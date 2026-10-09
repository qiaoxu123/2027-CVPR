import numpy as np
import torch

from graphnav.semantic_geometry import (
    build_entities, geometry_pair_distance, parse_spatial_relations,
    prior_and_rule_fields,
)
from graphnav.semantic_graph import prepare_semantic_graph, TypedGeometryGAT
from graphnav.visibility_graph import LandmarkVisibilityGraph


def _square(x, y, w=.05):
    return [[x, y], [x+w, y], [x+w, y+w], [x, y+w]]


def _landmarks():
    return [
        {"id":"7","name":"St Mary's Church","contour":_square(.20,.30)},
        {"id":"8","name":"Library","contour":_square(.55,.30)},
    ]


def test_building_is_one_node_and_road_is_one_polyline():
    road = [{"id":"r1","name":"Main Road",
             "polyline":[[.10,.40],[.5,.40],[.8,.40]], "osm_node_ids":["A","junction","B"]},
            {"id":"r2","name":"Cross Street",
             "polyline":[[.50,.20],[.5,.40],[.5,.70]], "osm_node_ids":["C","junction","D"]}]
    g = prepare_semantic_graph(_landmarks(),road,
        "east of St Mary's Church and along Main Road",
        [0,0,400,400], max_landmarks=3,max_roads=3,max_relations=3)
    assert len(g["entities"]) == 4
    assert len(g["relations"]) == 2
    assert g["node_mask"].sum() == 6
    assert g["node_types"][0] == 0 and g["node_types"][2] == 1
    assert g["adjacency"][2,3] and g["adjacency"][3,2]
    assert g["edges"][2,3,8+4] == 1  # actual connected roads
    assert g["adjacency"][4,0] and g["adjacency"][0,4]
    assert not g["adjacency"][4,1]  # language east node only attached to church
    assert g["edges"].shape[-1] == 15


def test_road_connectivity_is_not_nearest_neighbor():
    roads=[{"id":"r1","name":"East Road","polyline":[[.1,.2],[.2,.2]]},
           {"id":"r2","name":"West Road","polyline":[[.2,.4],[.3,.4]]}]
    g=prepare_semantic_graph([],roads,"", [0,0,400,400],
       max_landmarks=1,max_roads=2,max_relations=1)
    assert not g["adjacency"][0,1]
    assert not g["adjacency"][1,0]


def test_rules_follow_world_compass_and_not_nadir_heading():
    ents=build_entities(_landmarks(),[],[0,0,400,400])
    relations=parse_spatial_relations("Go east of St Mary's Church", ents)
    assert len(relations)==1 and relations[0].kind=="east"
    east=np.array([[280.,130.]],np.float32)
    west=np.array([[40.,130.]],np.float32)
    _,se,_=prior_and_rule_fields(east,ents,relations)
    _,sw,_=prior_and_rule_fields(west,ents,relations)
    assert se.item() > sw.item()


def test_near_uses_polygon_boundary_not_centroid():
    ents=build_entities(_landmarks(),[],[0,0,400,400])
    rel=parse_spatial_relations("near the Library",ents)
    assert len(rel)==1
    near=np.array([[220,140]],np.float32)
    far=np.array([[25,25]],np.float32)
    assert prior_and_rule_fields(near,ents,rel)[1][0] > prior_and_rule_fields(far,ents,rel)[1][0]


def test_between_two_identified_polygons_and_ambiguity_preserved():
    duplicate=_landmarks()+[{"id":"9","name":"Library","contour":_square(.65,.30)}]
    ents=build_entities(duplicate,[],[0,0,400,400])
    relations=parse_spatial_relations("between St Mary's Church and Library",ents)
    assert len(relations)==1 and relations[0].kind=="between"
    assert len(relations[0].anchors[1])==2
    mid=np.array([[150.,128.]],np.float32)
    away=np.array([[150.,380.]],np.float32)
    assert prior_and_rule_fields(mid,ents,relations)[1].item() > prior_and_rule_fields(away,ents,relations)[1].item()


def test_typed_edge_gat_backward_and_masks():
    torch.manual_seed(2)
    g=prepare_semantic_graph(_landmarks(),[],"near Library",
      [0,0,400,400], max_landmarks=3,max_roads=2,max_relations=2)
    e=TypedGeometryGAT(hidden=16,text_dim=8,heads=4,dropout=0)
    feature=torch.tensor(g["nodes"]).unsqueeze(0)
    names=torch.randn(1,len(g["node_mask"]),8)
    ty=torch.tensor(g["node_types"]).unsqueeze(0)
    adj=torch.tensor(g["adjacency"]).unsqueeze(0)
    edges=torch.tensor(g["edges"]).unsqueeze(0)
    valid=torch.tensor(g["node_mask"]).unsqueeze(0)
    out=e(feature,names,ty,adj,edges,valid)
    assert out.shape==(1,len(g["node_mask"]),16)
    assert out[0,~valid[0]].abs().sum()==0
    out.square().sum().backward()
    assert e.layer1.edge.weight.grad is not None


def test_visibility_needs_both_real_coverage_and_identity_match():
    polygon={"church":np.array([[40,40],[60,40],[60,60],[40,60]])}
    tracker=LandmarkVisibilityGraph(polygon,min_coverage=.30,min_match_confidence=.7)
    far=[[200,200],[220,200],[220,220],[200,220]]
    tracker.observe(far,{"church":1.0},observation_id="fake")
    assert tracker.evidence_for("church").observations==0
    assert tracker.edges["church"]==set()
    view=[[35,65],[65,65],[65,35],[35,35]]
    tracker.observe(view,{},observation_id="unverified")
    assert tracker.evidence_for("church").observations==0
    tracker.observe(view,{"church":.92},observation_id="verified")
    assert tracker.evidence_for("church").historical_match_confidence>.6
    assert tracker.edges["church"]=={"verified"}
