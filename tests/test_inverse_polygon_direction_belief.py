"""Synthetic inverse Polygon dataset + direction Belief network safeguards."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from graphnav.inverse_polygon_direction_data import (
    cardinal_from_whole_outlines,create_pair,build_dataset,INVERSE,
)
from graphnav.inverse_polygon_direction_model import (
    map_grid,polygon_target_distribution,collate_direction_features,
    DirectionalPolygonBelief,
)
from scripts.train_inverse_polygon_directions import read_quartets,quartet_loss


def entry(x0,y0,x1,y1,name=""):
    return dict(name=name,object_type="Building",
                contour=[[x0,y0],[x1,y0],[x1,y1],[x0,y1]])


def test_true_swap_and_wrong_word_keep_original_target_label():
    target=entry(65,41,75,51,"Target")
    anchor=entry(30,40,40,50,"One Stop")
    q=create_pair("test_map","test_map:1:0:0","1",target,
                  "2",anchor,[0,0,100,100])
    assert len(q)==4
    f,inv,wrong,wrong_inv=q
    assert f["input"]["relation_phrase"]=="map-right"
    assert inv["input"]["relation_phrase"]=="map-left"
    assert wrong["input"]["relation_phrase"]=="map-left"
    assert wrong_inv["input"]["relation_phrase"]=="map-right"
    assert f["label"]["truth"] is True
    assert wrong["label"]["truth"] is False
    assert wrong["label"]["target_contour"]==f["label"]["target_contour"]
    assert wrong["input"]["anchor_contour"]==f["input"]["anchor_contour"]
    assert inv["input"]["anchor_contour"]==f["label"]["target_contour"]
    assert inv["label"]["target_contour"]==f["input"]["anchor_contour"]
    for row in q:
        assert "target_contour" not in row["input"]


def test_diagonal_and_overlap_not_forced_direction():
    from shapely.geometry import Polygon
    a=Polygon(entry(0,0,10,10)["contour"])
    b=Polygon(entry(35,35,45,45)["contour"])
    assert cardinal_from_whole_outlines(b,a) is None
    assert cardinal_from_whole_outlines(a,a) is None


def test_polygon_target_label_and_network_with_belief_gradients():
    target=entry(65,41,75,51)
    anchor=entry(30,40,40,50,"One Stop")
    q=create_pair("m","m:1:0:0","1",target,"2",anchor,[0,0,100,100])
    batch=collate_direction_features(q,"cpu")
    assert batch["geometry"].shape==(4,784,8)
    assert batch["outline"].shape==(4,16,24,4)
    assert batch["soft_target"].shape==(4,784)
    torch.testing.assert_close(batch["soft_target"].sum(-1),torch.ones(4))
    net=DirectionalPolygonBelief(hidden=32,text_dim=16)
    bert=torch.randn(4,16)
    logits=net(batch["geometry"],batch["outline"],batch["part_mask"],
               batch["areas"],batch["direction_id"],bert[batch["direction_id"]])
    assert logits.shape==(4,784)
    objective,parts=quartet_loss(net,batch,bert)
    objective.backward()
    assert torch.isfinite(objective)
    assert torch.isfinite(net.direction.weight.grad).all()


def test_dataset_train_only_and_map_disjoint(tmp_path):
    records=[]
    catalog={}
    for mapname in ("alpha_block_0","beta_block_0"):
        records.append(dict(
            map_name=mapname,episode_id=f"{mapname}:1:0:0",
            instruction="Find a building by One Stop",
            landmarks=[dict(id="2",name="One Stop")],
            world_bounds=[0,0,100,100],
            target_xy=[.7,.5],
        ))
        catalog[mapname]={
            "1":entry(65,41,75,51,"T"),
            "2":entry(30,40,40,50,"One Stop"),
        }
    src=tmp_path/"train.jsonl"
    src.write_text("".join(json.dumps(r)+"\n" for r in records))
    obj=tmp_path/"objects.json";obj.write_text(json.dumps(catalog))
    out=tmp_path/"synth"
    m=build_dataset(src,obj,out)
    assert m["selected_pairs"]==2
    assert m["fit_samples"]==4 and m["dev_samples"]==4
    assert not set(m["fit_maps"])&set(m["dev_maps"])
    assert len(read_quartets(out/"fit.jsonl"))==1
    assert len(read_quartets(out/"dev.jsonl"))==1
    with pytest.raises(FileExistsError):
        build_dataset(src,obj,out)


def test_not_using_noisy_direction_words():
    anchor=entry(30,40,40,50,"One Stop")
    target=entry(65,41,75,51)
    q=create_pair("m","m:1:0:0","1",target,"2",anchor,[0,0,100,100])
    assert all(x["input"]["relation_phrase"] in INVERSE for x in q)
    assert not any("front" in x["input"]["instruction"] or
                   "north" in x["input"]["instruction"] for x in q)


def test_same_polygon_four_word_sweep_does_not_read_target_label():
    from scripts.visualize_inverse_direction_belief import run_sweep
    target=entry(65,41,75,51)
    anchor=entry(30,40,40,50,"One Stop")
    records=create_pair("m","m:1:0:0","1",target,"2",anchor,[0,0,100,100])
    row=dict(records[0])
    row.pop("label")  # forbidden field is absent
    model=DirectionalPolygonBelief(hidden=32,text_dim=16)
    bert=torch.randn(4,16)
    scores=run_sweep(model,bert,row,device="cpu")
    assert set(scores)=={"map-right","map-left","map-up","map-down"}
    for word,info in scores.items():
        assert info["probability"].shape==(28,28)
        assert info["probability"].sum()==pytest.approx(1,abs=1e-5)
        assert len(info["centroid_xy"])==2
