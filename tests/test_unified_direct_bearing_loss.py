"""Angular main-network Loss is direct, circular and STRICTLY GT-free at forward."""
import copy
import json
import math

import pytest
import torch

from graphnav.unified_direct_bearing_loss import (
    DirectBearingDataset,angular_consistency_loss,
    direct_cardinal_reference,predicate_ids_for_record,
)


def _catalog():
    return {
        "cambridge_1":{
            "1":dict(name="One Stop",object_type="Building",contour=[[10,10],[20,10],[20,20],[10,20]]),
            "2":dict(name="Different Store",object_type="Building",contour=[[60,50],[80,50],[80,70],[60,70]]),
        }
    }


def _row(instruction="The blue car north of One Stop",target=(.7,.6)):
    return dict(map_name="cambridge_1",instruction=instruction,
        target_xy=list(target),physical_map_sizes=[100.,100.],
        world_bounds=[0.,0.,100.,100.],
        dataset_version="revised",match_mode="fuzzy",
        landmarks=[
            dict(id="1",name="One Stop",object_type="Building",
                 contour=[[.10,.10],[.20,.10],[.20,.20],[.10,.20]])
        ])


def test_direct_world_cardinal_only_and_scope_guard():
    names={"one stop":{"1"}}
    reference={"id":"1","name":"One Stop"}
    assert direct_cardinal_reference("The blue car north of One Stop",reference,names)==3
    assert direct_cardinal_reference("The red truck east of One Stop",reference,names)==1
    assert direct_cardinal_reference("The red truck west of One Stop",reference,names)==2
    assert direct_cardinal_reference("The white building south of One Stop",reference,names)==4
    assert direct_cardinal_reference(
        "The blue car in the parking lot north of One Stop",reference,names)==0
    assert direct_cardinal_reference("The blue car behind One Stop",reference,names)==0
    assert direct_cardinal_reference("The blue car left of One Stop",reference,names)==0
    assert direct_cardinal_reference("The blue car north of One Stop and another car",reference,names)==0
    assert direct_cardinal_reference("The blue car north of One Stop",reference,
                                     {"one stop":{"1","5"}})==0


def test_anchor_selection_ignores_gt_and_rejects_same_name_multi_polygon():
    catalog=_catalog()["cambridge_1"]
    a=_row(target=(.9,.9))
    b=_row(target=(.1,.1))
    ids1,ind1=predicate_ids_for_record(a,catalog,36)
    ids2,ind2=predicate_ids_for_record(b,catalog,36)
    assert ind1==ind2==0
    assert torch.equal(ids1,ids2)
    assert ids1[:4].tolist()==[3,0,0,0]
    bad=copy.deepcopy(catalog)
    bad["99"]=dict(name="One Stop",object_type="Building")
    ids,index=predicate_ids_for_record(a,bad,36)
    assert index==-1
    assert ids.sum()==0
    a["landmarks"].append(dict(id="2",name="Different Store"))
    assert predicate_ids_for_record(a,catalog,36)[1]==-1


def test_dataset_appends_only_predicates_and_no_extra_gt_forward_inputs(tmp_path):
    catalog=tmp_path/"public.json"
    catalog.write_text(json.dumps(_catalog()))
    a=_row(target=(.7,.2))
    b=_row(target=(.3,.8))
    data=tmp_path/"all.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in (a,b))+"\n")
    ds=DirectBearingDataset(data,public_objects=catalog,deduplicate_queries=False)
    assert len(ds)==2
    assert ds.eligibility_audit["direct_cardinal_unique"]==2
    x=ds[0];y=ds[1]
    assert x["angular_anchor_index"].item()==y["angular_anchor_index"].item()==0
    assert torch.equal(x["anchor_predicate_ids"],y["anchor_predicate_ids"])
    assert x["anchor_predicate_ids"].shape==(36,)
    assert x["anchor_predicate_ids"][0].item()==3


def test_circular_loss_prefers_correct_side_and_gradients_flow_to_goal_belief():
    b=1;q=4;m=1
    pair=torch.zeros((b,q,m,10))
    pair[0,:,0,0:2]=torch.tensor([
        [0.10,0.0],[-0.10,0.0],[0.0,0.10],[0.0,-0.10],
    ])
    nodes=torch.zeros((b,m,10))
    nodes[0,0,:2]=torch.tensor([.5,.5])
    gt=torch.tensor([[.75,.5]])
    scale=torch.tensor([[100.,100.]])
    anchor=torch.tensor([0])
    right=torch.tensor([[5.,-3.,-3.,-3.]],requires_grad=True)
    wrong=torch.tensor([[-3.,5.,-3.,-3.]],requires_grad=True)
    good=angular_consistency_loss(right,pair,gt,nodes,scale,anchor)
    bad=angular_consistency_loss(wrong,pair,gt,nodes,scale,anchor)
    assert good["eligible_count"]==1
    assert bad["loss"]>good["loss"]
    good["loss"].backward()
    assert right.grad is not None
    assert float(right.grad.abs().sum())>0
    # West is the opposite circular class, not 8 bins away.
    assert good["gt_angle_bin"][0].item()==0


def test_zero_loss_on_no_direct_relation_and_no_geographic_yaw_assumption():
    logits=torch.randn(2,4,requires_grad=True)
    pair=torch.zeros(2,4,2,10)
    target=torch.tensor([[.6,.5],[.5,.5]])
    nodes=torch.zeros(2,2,10)
    scale=torch.ones(2,2)*100
    result=angular_consistency_loss(
        logits,pair,target,nodes,scale,torch.tensor([-1,-1]))
    assert result["eligible_count"]==0
    assert result["loss"].item()==0
    result["loss"].backward()
    assert logits.grad.abs().sum()==0


def test_gt_at_polygon_center_has_no_defined_bearing():
    logits=torch.zeros(1,4,requires_grad=True)
    pair=torch.zeros(1,4,1,10)
    center=torch.zeros(1,1,10)
    center[0,0,:2]=torch.tensor([.5,.5])
    out=angular_consistency_loss(logits,pair,torch.tensor([[.5,.5]]),
                                 center,torch.tensor([[100.,100.]]),torch.tensor([0]))
    assert out["eligible_count"]==0


def test_refuse_invalid_anchor_index():
    with pytest.raises(ValueError,match="out of bounds"):
        angular_consistency_loss(torch.zeros(1,4),torch.zeros(1,4,1,10),
            torch.zeros(1,2),torch.zeros(1,1,10),
            torch.ones(1,2)*100,torch.tensor([5]))
