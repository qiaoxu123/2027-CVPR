import torch
from graphnav.full_localization import GoalField
from graphnav.local_chain_runtime import local_goal_field


def base_field():
    axis=(torch.arange(4)+.5)/4
    y,x=torch.meshgrid(axis,axis,indexing='ij');xy=torch.stack((x.flatten(),y.flatten()),-1)[None]
    context=torch.cat((xy,xy),-1)
    ids=torch.tensor([[2,0,5,7]])
    logits=torch.zeros(1,16,dtype=torch.bfloat16)
    return GoalField(logits,ids,xy[:,ids[0]],context[:,ids[0]],dense_context=context),xy


def test_no_observation_keeps_exact_original_field():
    base,xy=base_field();uv=torch.tensor([[[.25,.5],[.5,.75]]])
    logits=torch.cat((base.belief_logits.float(),torch.full((1,2),-100.)),1)
    assert local_goal_field(base,xy,uv,logits,torch.zeros(1,2,dtype=torch.bool)) is base


def test_local_peak_has_matching_coordinate_and_interpolated_context():
    base,xy=base_field();uv=torch.tensor([[[.25,.5],[.5,.75]]])
    logits=torch.cat((base.belief_logits.float(),torch.tensor([[10.,-100.]])),1)
    field=local_goal_field(base,xy,uv,logits,torch.tensor([[True,False]]))
    assert field.belief_logits.shape==(1,17)
    assert field.candidate_ids[0,0]==16
    torch.testing.assert_close(field.candidate_xy[0,0],uv[0,0])
    torch.testing.assert_close(field.candidate_context[0,0,:2],uv[0,0])


def test_local_nonwinner_does_not_reorder_original_ties():
    base,xy=base_field();uv=torch.tensor([[[.25,.5]]])
    logits=torch.cat((base.belief_logits.float(),torch.tensor([[-100.]])),1)
    field=local_goal_field(base,xy,uv,logits,torch.ones(1,1,dtype=torch.bool))
    torch.testing.assert_close(field.candidate_ids,base.candidate_ids)
