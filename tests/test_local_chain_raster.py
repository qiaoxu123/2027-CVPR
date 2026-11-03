import torch
from graphnav.local_chain_raster import rasterized_goal_logprob, raster_goal_field
from test_local_chain_runtime import base_field


def test_mass_conserved_and_bilinear_world_distribution():
    joint=torch.cat((torch.full((1,16),-100.),torch.zeros(1,1)),1)
    uv=torch.tensor([[[.5,.5]]]);valid=torch.ones(1,1,dtype=torch.bool)
    probability=rasterized_goal_logprob(joint,uv,valid,global_cells=16).exp()
    torch.testing.assert_close(probability.sum(-1),torch.ones(1))
    torch.testing.assert_close(probability[0,torch.tensor([5,6,9,10])],torch.full((4,),.25))


def test_fixed_world_goal_remains_arrivable_as_patch_position_changes():
    base,xy=base_field();joint=torch.cat((torch.full((1,16),-10.),torch.tensor([[10.]])),1)
    valid=torch.ones(1,1,dtype=torch.bool)
    a=raster_goal_field(base,xy,torch.tensor([[[.625,.625]]]),joint,valid)
    b=raster_goal_field(base,xy,torch.tensor([[[.64,.63]]]),joint,valid)
    torch.testing.assert_close(a.candidate_xy[0,0],xy[0,10])
    torch.testing.assert_close(a.candidate_xy[0,0],b.candidate_xy[0,0])
    assert (a.candidate_xy[0,0]*400-b.candidate_xy[0,0]*400).norm()<=.001
    assert a.belief_logits.shape==(1,16)
    assert a.dense_context is base.dense_context


def test_no_observation_returns_original_field_and_tie_order():
    base,xy=base_field();uv=torch.tensor([[[.625,.625]]]);valid=torch.zeros(1,1,dtype=torch.bool)
    joint=torch.cat((base.belief_logits.float(),torch.tensor([[100.]])),1)
    assert raster_goal_field(base,xy,uv,joint,valid) is base


def test_raster_is_differentiable_for_end_to_end_goal_loss():
    joint=torch.randn(2,19,requires_grad=True);uv=torch.rand(2,3,2);valid=torch.ones(2,3,dtype=torch.bool)
    lp=rasterized_goal_logprob(joint,uv,valid,global_cells=16)
    (-lp[:,5].mean()).backward()
    assert torch.isfinite(joint.grad).all() and joint.grad[:,16:].norm()>0
