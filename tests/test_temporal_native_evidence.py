import torch
from graphnav.temporal_native_evidence import CausalResidualWindow


def test_history_averages_only_observed_support_and_expires_old_views():
    memory=CausalResidualWindow(2)
    first=memory.update(torch.tensor([[2.,99.,0.]]),torch.tensor([[True,False,False]]))
    assert torch.equal(first,torch.tensor([[2.,0.,0.]]))
    second=memory.update(torch.tensor([[0.,-2.,0.]]),torch.tensor([[False,True,False]]))
    assert torch.equal(second,torch.tensor([[2.,-2.,0.]]))
    third=memory.update(torch.tensor([[0.,2.,0.]]),torch.tensor([[False,True,False]]))
    assert torch.equal(third,torch.zeros(1,3))
    memory.reset()
    assert torch.equal(memory.update(torch.zeros(1,3),torch.zeros(1,3,dtype=torch.bool)),torch.zeros(1,3))


def test_one_view_is_identity_and_history_cannot_modify_earlier_outputs():
    one=CausalResidualWindow(1);mask=torch.ones(1,3,dtype=torch.bool);delta=torch.tensor([[.5,-1.,2.]])
    saved=one.update(delta,mask);assert torch.equal(saved,delta)
    one.update(-delta,mask)
    assert torch.equal(saved,delta)
