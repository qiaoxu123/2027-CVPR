import torch,pytest


def test_shift_kernel_obeys_east_north_grid_and_normalizes():
 from graphnav.direction_belief import apply_offset_kernel
 a=torch.zeros(1,28,28);a[0,10,10]=1
 # kernel indexed by target-minus-anchor offsets: +x is east, +y north.
 k=torch.zeros(1,55,55);k[0,28,29]=1
 p=apply_offset_kernel(a,k)
 assert torch.argmax(p).item()==11*28+12
 assert torch.allclose(p.sum(),torch.tensor(1.))


def test_relative_frame_requires_explicit_heading_and_gradients_flow():
 from graphnav.direction_belief import DirectionBelief
 m=DirectionBelief();a=torch.ones(2,28,28)/784;span=torch.tensor([[400.,400.]]).repeat(2,1)
 with pytest.raises(ValueError):m(a,torch.tensor([4,5]),span)
 p=m(a,torch.tensor([0,1]),span);assert p.shape==(2,28,28) and torch.isfinite(p).all()
 loss=-p[:,20,14].clamp_min(1e-20).log().mean();loss.backward()
 assert any(x.grad is not None and x.grad.abs().sum()>0 for x in m.parameters())


def test_known_heading_rotates_relative_kernel_only():
 from graphnav.direction_belief import rotate_means
 means=torch.tensor([[50.,0.],[50.,0.]])
 result=rotate_means(means,torch.tensor([4,2]),torch.tensor([torch.pi/2,torch.pi/2]))
 torch.testing.assert_close(result,torch.tensor([[0.,50.],[50.,0.]]),atol=1e-4,rtol=0)


def test_optimizer_resume_matches_continuous_training():
 import copy
 from graphnav.direction_belief import DirectionBelief
 a=torch.zeros(1,28,28);a[0,14,14]=1;ids=torch.tensor([0]);span=torch.tensor([[400.,400.]])
 model=DirectionBelief();opt=torch.optim.Adam(model.parameters(),lr=.01)
 def step(m,o):
  loss=-m(a,ids,span)[0,18,14].clamp_min(1e-20).log();o.zero_grad();loss.backward();o.step()
 step(model,opt);weights=copy.deepcopy(model.state_dict());state=copy.deepcopy(opt.state_dict());step(model,opt)
 restored=DirectionBelief();restored.load_state_dict(weights);resumed=torch.optim.Adam(restored.parameters(),lr=.01);resumed.load_state_dict(state);step(restored,resumed)
 for x,y in zip(model.parameters(),restored.parameters()):torch.testing.assert_close(x,y,atol=0,rtol=0)
 with pytest.raises(TypeError):restored(a,ids,span,target_world=torch.zeros(1,2))
