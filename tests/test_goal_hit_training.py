import torch
from graphnav.goal_hit_training import goal_hit_loss

def test_hit_loss_gradients_raise_good_region_without_inference_labels():
    z=torch.tensor([[0.,2.,0.]],requires_grad=True);grid=torch.tensor([[[.1,.1],[.7,.7],[.8,.8]]]);bounds=torch.tensor([[0.,0.,100.,100.]]);target=torch.tensor([[10.,10.]]);loss,parts=goal_hit_loss(z,grid,bounds,target,target/100,'hit20');loss.backward()
    assert torch.isfinite(loss) and z.grad[0,0]<0 and z.grad[0,1]>0
    assert parts['no_positive20']==0

def test_outside_grid_target_does_not_create_fake_hit_or_nan():
    z=torch.zeros(1,3,requires_grad=True);grid=torch.tensor([[[0.,0.],[.5,.5],[1.,1.]]]);bounds=torch.tensor([[0.,0.,100.,100.]]);target=torch.tensor([[300.,300.]])
    loss,parts=goal_hit_loss(z,grid,bounds,target,target/100,'hit20');loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all() and parts['no_positive20']==1

def test_mixed_covered_and_uncovered_targets_have_finite_gradients():
    z=torch.zeros(2,3,requires_grad=True);grid=torch.tensor([[[0.,0.],[.5,.5],[1.,1.]]]).repeat(2,1,1);bounds=torch.tensor([[0.,0.,100.,100.]]).repeat(2,1);target=torch.tensor([[0.,0.],[300.,300.]])
    loss,parts=goal_hit_loss(z,grid,bounds,target,target/100,'hit20');loss.backward()
    assert parts['no_positive20']==1 and torch.isfinite(z.grad).all()
