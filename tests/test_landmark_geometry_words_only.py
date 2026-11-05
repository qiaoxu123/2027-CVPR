"""Strict minimal language inputs and independent PUBLIC polygon tests."""
import json

import numpy as np
import pytest
import torch

from graphnav.landmark_geometry_words import extract_landmark_geometry
from graphnav.landmark_geometry_only_model import LandmarkGeometryWordsOnly
from scripts.train_landmark_geometry_words import (
    LandmarkWordsDataset,collate,encode_model_inputs,
)


def _ref(ident,name,contour=None):
    return dict(
        id=ident,name=name,aliases=[],
        contour=contour or [[.15,.15],[.35,.15],[.35,.35],[.15,.35]],
    )


def test_extract_only_landmark_name_and_adjacent_geometry_not_target_description():
    text="The blue car in the parking lot behind One Stop, next to a black car."
    parsed=extract_landmark_geometry(text,[_ref("A","One Stop")])
    assert parsed["matched_names"]==1
    assert parsed["paired_geometry"]==1
    match=parsed["anchors"][0]
    assert match["name"]=="One Stop"
    assert match["geometry_words"]=="behind"
    assert all(k not in match for k in ("car","target","color","instruction","parking"))


def test_multiple_names_independent_and_unrelated_words_not_assigned():
    text="A blue car north of One Stop and next to Blackwell Street."
    data=extract_landmark_geometry(
        text,[_ref("A","One Stop"),_ref("B","Blackwell Street")]
    )
    assert [(r["name"],r["geometry_words"]) for r in data["anchors"]]==[
        ("One Stop","north of"),("Blackwell Street","next to")
    ]
    assert data["paired_geometry"]==2


def test_no_adjacent_geometry_word_yields_name_only_not_nearest_keyword():
    text="The north parking lot is far away from One Stop."
    data=extract_landmark_geometry(text,[_ref("A","One Stop")])
    assert data["anchors"][0]["geometry_words"]==""
    assert data["paired_geometry"]==0


def test_preserve_all_homonymous_public_polygon_parts():
    text="The car near One Stop."
    data=extract_landmark_geometry(text,[
        _ref("1","One Stop"),
        _ref("2","One Stop",contour=[[.5,.5],[.7,.5],[.7,.7],[.5,.7]])
    ])
    assert len(data["anchors"])==2
    assert {r["public_id"] for r in data["anchors"]}=={"1","2"}
    assert all(x["geometry_words"]=="near" for x in data["anchors"])


def test_longer_landmark_name_wins_over_shorter_overlap():
    text="The car behind North Street Bridge."
    data=extract_landmark_geometry(text,[
        _ref("a","North Street"),_ref("b","North Street Bridge")])
    assert len(data["anchors"])==1
    assert data["anchors"][0]["name"]=="North Street Bridge"
    assert data["anchors"][0]["geometry_words"]=="behind"


def test_unknown_reference_frame_word_not_forced_to_map_north():
    text="The vehicle right of One Stop."
    data=extract_landmark_geometry(text,[_ref("a","One Stop")])
    assert data["anchors"][0]["geometry_words"]=="right of"


def test_full_public_catalog_used_without_gt_selected_landmarks(tmp_path):
    catalog=dict(
        cambridge_block_1=dict(
            sourceA=dict(name="One Stop",object_type="Building",
                         contour=[[10,10],[30,10],[30,30],[10,30]])),
        birmingham_block_1=dict(
            sourceB=dict(name="South Gate",object_type="Building",
                         contour=[[10,10],[30,10],[30,30],[10,30]])))
    public=tmp_path/"objects.json"
    public.write_text(json.dumps(catalog))
    rows=[
        dict(map_name="cambridge_block_1",world_bounds=[0,0,100,100],
             instruction="The blue car behind One Stop.",
             landmarks=[], # pre-selected refs deliberately EMPTY
             target_xy=[.5,.5],physical_map_sizes=[100,100],
             dataset_version="revised",match_mode="fuzzy"),
        dict(map_name="birmingham_block_1",world_bounds=[0,0,100,100],
             instruction="Find the green building.",
             landmarks=[dict(name="South Gate",contour=[[0,0],[1,0],[1,1],[0,1]])],
             target_xy=[.5,.5],physical_map_sizes=[100,100],
             dataset_version="revised",match_mode="fuzzy"),
    ]
    train=tmp_path/"input.jsonl"
    train.write_text("".join(json.dumps(x)+"\n" for x in rows))
    dataset=LandmarkWordsDataset(train,public)
    assert len(dataset)==2
    assert dataset.counter["paired"]==1
    assert dataset.counter["no_public_name"]==1
    x=dataset[0]["model_inputs"]
    assert x["name_text"][0]=="One Stop"
    assert x["geometry_words"][0]=="behind"
    assert torch.allclose(x["contours"][0,0],torch.tensor([.1,.1]))
    assert dataset[1]["model_inputs"]["anchor_mask"].sum()==0
    a=collate([dataset[0],dataset[1]])
    assert "instruction" not in a["model_inputs"]
    assert "target_xy" not in a["model_inputs"]
    assert a["label_xy"].shape==(2,2)


def test_model_unknown_public_name_returns_uniform_no_target_hallucination():
    model=LandmarkGeometryWordsOnly()
    torch.manual_seed(17)
    b=2;m=16
    contours=torch.zeros(b,m,16,2)
    contours[0,0]=torch.tensor([[.2,.2],[.4,.2],[.4,.4],[.2,.4]]).repeat(4,1)
    valid=torch.zeros(b,m,dtype=torch.bool);valid[0,0]=True
    words=valid.clone()
    from graphnav.geometry import grid_centers
    grid=grid_centers(28)[None].expand(b,-1,-1)
    x=dict(name_embedding=torch.randn(b,m,768),
           words_embedding=torch.randn(b,m,768),
           landmark_contours=contours,landmark_valid=valid,
           words_valid=words,candidate_xy=grid)
    logits=model(**x)
    assert logits.shape==(2,784)
    assert torch.equal(logits[1],torch.zeros(784))
    assert torch.isfinite(logits).all()
    loss=torch.nn.functional.cross_entropy(logits[0:1],torch.tensor([250]))
    loss.backward()
    assert model.merge[-1].weight.grad is not None
    assert model.merge[-1].weight.grad.abs().sum()>0
    with pytest.raises(TypeError):
        model(**x,target_xy=torch.tensor([[.5,.5],[.5,.5]]))
