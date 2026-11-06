"""CPU tests: ONLY direction as new modality; no GT or radial magnitude."""
import math

import pytest
import torch

from graphnav.full_localization import GoalField
from graphnav.unified_angle_only_belief import (
    AngleOnlyBelief,bearing_supervision,named_anchor_mask,public_candidate_bearings,
)
from graphnav.unified_angle_only_runtime import revised_field_with_angles


def batch():
    torch.manual_seed(42)
    b=2;q=16;n=3;length=8
    pair=torch.randn(b,q,n,10)
    name=torch.randn(b,n,768)
    return dict(
        base_logits=torch.randn(b,q),
        candidate_bearings=public_candidate_bearings(pair),
        landmark_valid=torch.tensor([[True,True,False],[False,False,False]]),
        tokens=torch.randn(b,length,768),
        token_mask=torch.tensor([[True]*8,[True]*6+[False]*2]),
        node_emb=name,
    )


def test_angle_only_cannot_observe_distance():
    pair=torch.randn(1,8,3,10)
    other=pair.clone()
    other[...,0:2]*=4.1
    other[...,2:10]*=19
    torch.testing.assert_close(public_candidate_bearings(pair),
                               public_candidate_bearings(other),atol=1e-6,rtol=1e-6)


def test_angle_bins_match_planar_x_axis_not_geographic_north():
    center=torch.tensor([[[0.,0.],[.5,.5],[0.,0.]]])
    target=torch.tensor([[.75,.5]])
    size=torch.tensor([[100.,100.]])
    mask=torch.tensor([[True,True,False]])
    labels,valid=bearing_supervision(target,center,size,mask,bins=8)
    assert labels[0,1].item()==0  # +X = 0 degrees
    assert labels[0,0].item()==1  # 45 degrees when physical dx=75,dy=50? -> 33.7 degrees rounds to 45.
    assert valid.tolist()==[[True,True,False]]
    vertical,_=bearing_supervision(torch.tensor([[.5,.75]]),
                                   torch.tensor([[[.5,.5]]]),
                                   size,torch.tensor([[True]]),bins=8)
    assert vertical.item()==2  # map +Y = 90 deg


def test_model_zero_initialization_and_unmatched_row_are_exact_baseline():
    x=batch()
    head=AngleOnlyBelief()
    y=head(**x)
    assert y["angle_logits"].shape==(2,3,8)
    assert torch.equal(y["logits"],x["base_logits"])
    assert torch.equal(y["delta"],torch.zeros_like(x["base_logits"]))
    assert y["supported"].tolist()==[True,False]
    with torch.no_grad():head.fusion_gain.copy_(torch.tensor(10.))
    y=head(**x)
    assert (y["delta"].abs()<=1.0000001).all()
    assert torch.equal(y["logits"][1],x["base_logits"][1])


def test_direction_supervision_gradients_and_goal_gradients():
    x=batch()
    head=AngleOnlyBelief()
    with torch.no_grad():head.fusion_gain.fill_(1.)
    y=head(**x)
    target=torch.tensor([0,3])
    direction_target=torch.tensor([0,3])
    loss=torch.nn.functional.cross_entropy(y["logits"],target)
    loss=loss+torch.nn.functional.cross_entropy(
        y["angle_logits"][0,:2],direction_target)
    loss.backward()
    assert head.direction_head[-1].weight.grad.abs().sum()>0
    assert head.fusion_gain.grad is not None
    with pytest.raises(TypeError):
        head(**x,target_xy=torch.zeros(2,2))
    with pytest.raises(TypeError):
        head.predict_angle(tokens=x["tokens"],
            token_mask=x["token_mask"],node_emb=x["node_emb"],
            landmark_valid=x["landmark_valid"],target_xy=torch.zeros(2,2))


def test_named_anchor_mask_excludes_public_roads_and_non_landmarks():
    entity_valid=torch.tensor([[True,True,True,False],[True,True,False,False]])
    mask=named_anchor_mask(entity_valid,torch.tensor([2,0]))
    assert mask.tolist()==[[True,True,False,False],[False]*4]


def test_bearing_class_response_changes_angular_field():
    head=AngleOnlyBelief(bins=8)
    with torch.no_grad():head.fusion_gain.fill_(8.)
    angle=torch.tensor([[[0.],[math.pi/2],[math.pi],[3*math.pi/2]]])
    known=torch.ones(1,1,dtype=torch.bool)
    original=torch.zeros(1,4)
    east=torch.nn.functional.one_hot(torch.tensor([[0]]),num_classes=8).float()
    west=torch.nn.functional.one_hot(torch.tensor([[4]]),num_classes=8).float()
    a=head.apply_angle(base_logits=original,candidate_bearings=angle,
                       landmark_valid=known,bearing_probabilities=east)
    b=head.apply_angle(base_logits=original,candidate_bearings=angle,
                       landmark_valid=known,bearing_probabilities=west)
    assert a["logits"][0].argmax().item()==0
    assert b["logits"][0].argmax().item()==2


def test_angular_runtime_reranks_xy_and_context_without_changing_stop_head():
    grid=torch.tensor([[[.1,.1],[.2,.2],[.3,.3],[.4,.4]]])
    dense=torch.arange(4*128,dtype=torch.float32).reshape(1,4,128)
    old=GoalField(
        belief_logits=torch.tensor([[3.,2.,1.,0.]]),
        candidate_ids=torch.tensor([[0,1]]),
        candidate_xy=grid[:,:2],candidate_context=dense[:,:2],
        dense_context=dense)
    newer=revised_field_with_angles(old,grid,torch.tensor([[0.,3.,1.,4.]]))
    assert newer.candidate_ids.tolist()==[[3,1]]
    torch.testing.assert_close(newer.candidate_xy,grid[:,[3,1]])
    torch.testing.assert_close(newer.candidate_context,dense[:,[3,1]])
    assert old.candidate_ids.tolist()==[[0,1]]
