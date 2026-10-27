import numpy as np
import torch
from experiments.a2_waldo_landmark_belief.observed_geometry import sample_surface, local_support, gaussian_belief_mix


def test_invalid_surface_stays_unknown_and_nonfinite_projection_is_rejected():
    height=np.ones((20,20),np.float32)*7
    polygon=np.array([[2,2],[17,2],[17,17],[2,17]],np.float32)
    assert sample_surface(height,np.ones_like(height,bool),polygon)['z']==7
    assert sample_surface(height,np.zeros_like(height,bool),polygon)['z'] is None
    assert sample_surface(height,np.ones_like(height,bool),polygon*np.nan)['z'] is None


def test_support_is_observed_and_candidate_local():
    grid=np.array([[0,0],[4,0],[40,0]],np.float32)
    polygon=np.array([[-1,-1],[1,-1],[1,1],[-1,1]],np.float32)
    support=local_support(grid,[polygon],np.array([False,True,True]))
    assert not support[0,0] and not support[0,2]


def test_gaussian_update_conserves_mass_and_has_offset_gradients():
    base=torch.tensor([-.3,-1.,-.5,-.8])
    grid=torch.tensor([[0.,0.],[3.,0.],[6.,0.],[20.,0.]])
    xy=torch.tensor([[2.,0.]],requires_grad=True)
    support=torch.tensor([[True,True,True,False]])
    logits=gaussian_belief_mix(base,grid,xy,torch.tensor([1.]),support,torch.tensor([True]),.5)
    assert logits[-1]==base[-1]
    assert torch.allclose(logits.exp().sum(),base.exp().sum(),atol=1e-6)
    logits[0].backward();assert xy.grad.abs().sum()>0
    assert gaussian_belief_mix(base,grid,xy,torch.tensor([1.]),support,torch.tensor([True]),0) is base
