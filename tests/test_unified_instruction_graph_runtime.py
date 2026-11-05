"""CPU safety for optional true closed-loop graph runtime adapter."""
from types import SimpleNamespace

import pytest
import torch

from graphnav.full_localization import GoalField
from graphnav.unified_instruction_graph_runtime import (
    attach_public_ids,replace_goal_field_logits,
)


def test_public_ids_restored_only_in_graph_sidecar():
    raw={}
    raw["42"]=dict(name="One Stop",object_type="Building",
                   contour=[[0,0],[20,0],[20,20],[0,20]])
    catalog=SimpleNamespace(objects={"cambridge_block_1":raw})
    episode=SimpleNamespace(map_name="cambridge_block_1",
                            reference_names=("One Stop",),references=())
    original=dict(map_name="cambridge_block_1",
                  instruction="The blue car behind One Stop",
                  landmarks=[dict(name="One Stop",object_type="Building",
                                  contour=[[0,0],[.2,0],[.2,.2],[0,.2]])])
    found=attach_public_ids(original,episode,catalog,16)
    assert found["landmarks"][0]["id"]=="42"
    assert "id" not in original["landmarks"][0]
    assert found["landmarks"][0]["contour"]==original["landmarks"][0]["contour"]


def test_changed_reference_order_refuses_silent_binding():
    raw={"42":dict(name="One Stop",object_type="Building",
                   contour=[[0,0],[20,0],[20,20],[0,20]])}
    catalog=SimpleNamespace(objects={"cambridge_block_1":raw})
    episode=SimpleNamespace(map_name="cambridge_block_1",
                            reference_names=("One Stop",),references=())
    bad=dict(landmarks=[dict(name="Unrelated",object_type="Building")])
    with pytest.raises(ValueError,match="order"):
        attach_public_ids(bad,episode,catalog,16)


def test_goal_field_topk_context_and_xy_change_together():
    grid=torch.tensor([[[.1,.1],[.2,.2],[.3,.3],[.4,.4]]])
    dense=torch.arange(1*4*128,dtype=torch.float32).reshape(1,4,128)
    old=GoalField(
        belief_logits=torch.tensor([[3.,2.,1.,0.]]),
        candidate_ids=torch.tensor([[0,1]]),
        candidate_xy=grid[:,:2],
        candidate_context=dense[:,:2],
        dense_context=dense,
    )
    changed=replace_goal_field_logits(old,grid,torch.tensor([[0.,3.,1.,4.]]))
    assert changed.candidate_ids.tolist()==[[3,1]]
    torch.testing.assert_close(changed.candidate_xy,grid[:,[3,1]])
    torch.testing.assert_close(changed.candidate_context,dense[:,[3,1]])
    assert old.candidate_ids.tolist()==[[0,1]]


def test_goal_field_grid_mismatch_refuses_invalid_update():
    grid=torch.zeros(1,4,2)
    dense=torch.zeros(1,4,128)
    old=GoalField(torch.zeros(1,4),torch.tensor([[0,1]]),grid[:,:2],
                  dense[:,:2],dense_context=dense)
    with pytest.raises(ValueError,match="mismatched"):
        replace_goal_field_logits(old,grid,torch.zeros(1,3))
