"""CPU: public-only capacity and Polygon repair preserve the frozen 16 slots."""
import copy
import pytest
import torch

from graphnav.public_anchor_repair import (
    conservative_repair_contour,repaired_public_text_record,
    select_complete_groups,sanitize_public_map,
)
from graphnav.unified_text_ablation import public_text_record
from scripts.analyze_instruction_landmark_association import PublicNames


def rectangle(x):
    return [[x,10],[x+1,10],[x+1,11],[x,11]]


def catalog(a=12,b=6):
    objects={}
    for i in range(a):
        objects[str(i)]=dict(name="One Stop",object_type="Building",
                            contour=rectangle(10+i*2))
    for i in range(b):
        objects[str(100+i)]=dict(name="Other School",object_type="Building",
                                contour=rectangle(100+i*2))
    return {"city_block_1":objects}


def bind(world, text, **kwargs):
    return repaired_public_text_record(
        "city_block_1",text,[0,0,200,200],world,PublicNames(world),
        mode="full_text",max_landmarks=16,**kwargs)


def test_unchanged_when_public_binding_within_capacity():
    world=catalog(3,2)
    text="The car beside One Stop and Other School"
    original,_=public_text_record("city_block_1",text,[0,0,200,200],
        world,PublicNames(world),mode="full_text",max_landmarks=16)
    corrected,audit=bind(world,text)
    assert corrected==original
    assert not audit["still_unbound"]
    assert audit["public_polygon_selected"]==5
    assert not audit["repaired_public_ids"]


def test_overflow_recovery_does_not_use_GT_or_truncate_same_name_group():
    world=catalog(12,6)
    text="The target in front of One Stop and Other School"
    old,_=public_text_record("city_block_1",text,[0,0,200,200],
        world,PublicNames(world),mode="full_text",max_landmarks=16)
    assert not old["landmarks"]
    fixed,audit=bind(world,text,repair_geometry=False)
    assert len(fixed["landmarks"])==12
    assert all(x["name"]=="One Stop" for x in fixed["landmarks"])
    assert audit["truncated_group_count"]==1
    assert audit["GT_used_to_bind"] is False
    assert audit["distance_used_to_select"] is False
    assert len(set(x["id"] for x in fixed["landmarks"]))==12
    with pytest.raises(TypeError):
        bind(world,text,target_xy=[.1,.2])


def test_single_oversize_homonym_group_remains_unsupported_not_partially_bound():
    world=catalog(19,0)
    record,audit=bind(world,"The car beside One Stop")
    assert record["landmarks"]==[]
    assert audit["still_unbound"]
    assert audit["complete_group_events"][0]["reason"]=="single_group_over_capacity"


def test_overlapping_alias_group_never_partially_selected():
    references=[dict(id=str(i)) for i in range(21)]
    mentions=[
        dict(text="Alpha",public_ids=[str(i) for i in range(12)]),
        dict(text="Alias",public_ids=[str(i) for i in range(5,18)]),
        dict(text="Other",public_ids=["18","19","20"]),
    ]
    selected,audit=select_complete_groups(mentions,references,capacity=16)
    assert [int(x["id"]) for x in selected]==[18,19,20]
    assert any(x["reason"]=="single_group_over_capacity" for x in audit)


def test_no_name_no_gt_fallback_and_original_catalog_unchanged():
    world=catalog(2,2)
    original=copy.deepcopy(world)
    record,audit=bind(world,"The blue car beside a white van")
    assert record["landmarks"]==[]
    assert audit["still_unbound"] and audit["public_name_mentions"]==0
    assert world==original


def test_simple_invalid_geometry_not_silently_fabricated():
    malformed=[[0,0],[2,2],[0,2],[2,0]]
    assert conservative_repair_contour(malformed) is None
    assert conservative_repair_contour([[0,0],[1,1]]) is None
    valid=rectangle(5)
    assert conservative_repair_contour(valid)==valid
    world=catalog(1,0)
    world["city_block_1"]["0"]["contour"]=malformed
    original=copy.deepcopy(world)
    updated,ids=sanitize_public_map(world["city_block_1"])
    assert updated==world["city_block_1"]
    assert not ids
    assert world==original
