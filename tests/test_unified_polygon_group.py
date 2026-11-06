"""CPU tests: no GT polygon selection, multi-ID group OR, map/cardinal safety."""
import copy
import math

import pytest
import torch

from graphnav.unified_polygon_group import public_cardinal_group,grouped_cardinal_loss
from scripts.analyze_instruction_landmark_association import PublicNames
from scripts.audit_unified_public_binding import inspect_rows


def _public():
    return {
        "city_block_1":{
            "11":dict(name="One Stop",object_type="Building",
                contour=[[10,10],[15,10],[15,15],[10,15]]),
            "12":dict(name="One Stop",object_type="Building",
                contour=[[70,30],[75,30],[75,35],[70,35]]),
            "13":dict(name="Other House",object_type="Building",
                contour=[[40,45],[45,45],[45,50],[40,50]]),
        }
    }


def _record(instruction="The blue car north of One Stop"):
    refs=[dict(id="11",name="One Stop",object_type="Building"),
          dict(id="12",name="One Stop",object_type="Building"),
          dict(id="13",name="Other House",object_type="Building")]
    return dict(map_name="city_block_1",instruction=instruction,
                landmarks=refs,target_xy=[.8,.3],
                dataset_version="revised",match_mode="fuzzy",
                world_bounds=[0.,0.,100.,100.])


def _bind(record,public=None):
    public=public or _public()
    return public_cardinal_group(
        record,PublicNames(public),public,36)


def test_duplicate_name_retained_as_TWO_individual_public_geometry_anchors():
    ids,mask,status=_bind(_record())
    assert status=="same_name_multi_polygon"
    assert ids[:4].tolist()==[3,3,0,0]
    assert mask[:4].tolist()==[True,True,False,False]


def test_gt_target_does_not_control_reference_selection_or_group_cardinal_ids():
    record=_record()
    ref=_bind(record)
    for xy in ([.1,.1],[.9,.9],[.4,.5]):
        new=copy.deepcopy(record);new["target_xy"]=xy
        ids,mask,status=_bind(new)
        assert status==ref[2]
        torch.testing.assert_close(ids,ref[0])
        torch.testing.assert_close(mask,ref[1])
    # A missing fragment must never be silently accepted as the complete group.
    removed=copy.deepcopy(record);removed["landmarks"].pop(1)
    ids,mask,status=_bind(removed)
    assert status=="no_conservative_direct_cardinal"
    assert ids.sum()==0 and mask.sum()==0


def test_direct_scope_and_unknown_frame_are_not_forced_to_world_north():
    for sentence in (
        "The blue car in the parking lot north of One Stop",
        "The blue car right of One Stop",
        "The blue car behind One Stop",
        "The blue car north of One Stop and another car",
    ):
        ids,mask,status=_bind(_record(sentence))
        assert status=="no_conservative_direct_cardinal"
        assert not mask.any()


def test_same_name_different_public_object_types_is_ambiguous():
    public=_public()
    public["city_block_1"]["12"]["object_type"]="TrafficRoad"
    ids,mask,status=_bind(_record(),public)
    assert not mask.any()
    assert status=="no_conservative_direct_cardinal"


def _one_or_two_group_loss(logits):
    # Four candidates: both north, one north, neither north, south.
    # This demonstrates the group union is NOT an average of two bearings.
    pair=torch.zeros(1,4,2,10)
    pair[0,:,:,0:2]=torch.tensor([
        [[0.,-1.],[0., 1.]],
        [[0., 1.],[0.,-1.]],
        [[0.,-1.],[0.,-1.]],
        [[1.,-1.],[1.,-1.]],
    ])
    group=torch.tensor([[True,True]])
    ids=torch.tensor([[3,3]])
    return grouped_cardinal_loss(logits,pair,group,ids)


def test_set_or_satisfies_ANY_polygon_and_has_belief_gradient():
    # q0 north of second Polygon but south of first; q1 vice versa.
    # Either must be accepted, without ever choosing a GT-nearest Polygon.
    good=torch.tensor([[4.,-5.,-5.,-5.]],requires_grad=True)
    bad=torch.tensor([[-5.,-5.,4.,-5.]],requires_grad=True)
    a=_one_or_two_group_loss(good)
    b=_one_or_two_group_loss(bad)
    assert a["eligible_count"]==1
    assert a["loss"]<b["loss"]
    a["loss"].backward()
    assert good.grad is not None
    assert float(good.grad.abs().sum())>0.


def test_set_or_is_permutation_invariant_over_same_name_poly_instances():
    logits=torch.randn(1,4)
    pair=torch.randn(1,4,3,10)
    ids=torch.tensor([[3,3,0]])
    mask=torch.tensor([[True,True,False]])
    score1=grouped_cardinal_loss(logits,pair,mask,ids)
    perm=[1,0,2]
    score2=grouped_cardinal_loss(logits,pair[:,:,perm,:],
                                mask[:,perm],ids[:,perm])
    torch.testing.assert_close(score1["loss"],score2["loss"],rtol=0,atol=1e-7)
    torch.testing.assert_close(score1["supported_mass"],score2["supported_mass"],
                               rtol=0,atol=1e-7)


def test_direction_swapping_changes_cardinal_constraints():
    pair=torch.zeros(1,2,1,10)
    pair[0,:,0,0:2]=torch.tensor([[0.,1.],[0.,-1.]])
    mask=torch.ones(1,1,dtype=torch.bool)
    logits=torch.tensor([[4.,-4.]])
    north=grouped_cardinal_loss(logits,pair,mask,torch.tensor([[3]]))
    south=grouped_cardinal_loss(logits,pair,mask,torch.tensor([[4]]))
    assert north["loss"]<south["loss"]


def test_empty_mask_gives_exact_zero_loss_and_no_GT_input():
    logits=torch.randn(2,6,requires_grad=True)
    pair=torch.randn(2,6,3,10)
    group=torch.zeros(2,3,dtype=torch.bool)
    ids=torch.zeros(2,3,dtype=torch.long)
    out=grouped_cardinal_loss(logits,pair,group,ids)
    assert out["eligible_count"]==0
    assert out["loss"].item()==0.
    out["loss"].backward()
    assert float(logits.grad.abs().sum())==0.
    with pytest.raises(TypeError):
        grouped_cardinal_loss(logits,pair,group,ids,target_xy=torch.zeros(2,2))


def test_audit_detects_source_annotation_reference_shift():
    public=_public()
    row=_record()
    # Pretend original annotation-indexed processed_descriptions selected
    # only one ID; PUBLIC-BLIND must recover BOTH from same instruction.
    row["landmarks"]=row["landmarks"][:1]
    count,details,examples,blind=inspect_rows([row],public,max_landmarks=16)
    assert count["queries"]==1
    assert count["blind_only_polygon_ids"]>=1
    assert {x["id"] for x in blind[0]["landmarks"]}=={"11","12"}
    assert blind[0]["target_xy"]==row["target_xy"]
    assert blind[0]["instruction"]==row["instruction"]
    assert len(blind[0]["landmarks"])==2


def test_offline_public_blind_binder_matches_REAL_online_full_text_adapter():
    from graphnav.unified_text_ablation import public_text_record
    catalog=_public()
    row=_record()
    _,_,_,blind=inspect_rows([row],catalog,16)
    online,audit=public_text_record(
        row["map_name"],row["instruction"],row["world_bounds"],
        catalog,PublicNames(catalog),mode="full_text",max_landmarks=16)
    assert [x["id"] for x in blind[0]["landmarks"]] == [
        x["id"] for x in online["landmarks"]
    ]
    assert [x["contour"] for x in blind[0]["landmarks"]] == [
        x["contour"] for x in online["landmarks"]
    ]
    assert online["instruction"]==row["instruction"]
    assert not audit["GT_input"]


def test_public_overcapacity_retains_query_but_refuses_partial_group():
    from graphnav.unified_text_ablation import public_text_record
    catalog=_public()
    for i in range(20,38):
        catalog["city_block_1"][str(i)]=dict(
            name="One Stop",object_type="Building",
            contour=[[i,50],[i+.4,50],[i+.4,51],[i,51]],
        )
    row=_record()
    online,audit=public_text_record(
        row["map_name"],row["instruction"],row["world_bounds"],
        catalog,PublicNames(catalog),mode="full_text",max_landmarks=16)
    assert len(online["landmarks"])==0
    assert "all_public_polygons_exceed_capacity" in audit["warnings"]
    ids,mask,status=public_cardinal_group(
        online,PublicNames(catalog),catalog,36)
    assert status=="no_public_anchor"
    assert not mask.any() and ids.sum()==0
