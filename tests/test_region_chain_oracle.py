import pytest
import torch
from graphnav.region_chain_oracle import oracle_region_rejection


def test_oracle_requires_explicit_diagnostic_mode():
    args=(torch.zeros(1,2),torch.zeros(1,2),torch.tensor([0]),torch.tensor([True]),2.,1)
    with pytest.raises(ValueError):oracle_region_rejection(*args)


def test_unknown_and_unobserved_cells_never_become_negative_evidence():
    world=torch.tensor([[0.,0.],[.5,0.],[10.,0.]])
    local=torch.tensor([[0.,0.]])
    result=oracle_region_rejection(world,local,torch.tensor([0]),torch.tensor([True]),2.,1,diagnostic=True)
    assert result.tolist()==[True,True,False]
    result=oracle_region_rejection(world,local,torch.tensor([-1]),torch.tensor([True]),2.,1,diagnostic=True)
    assert not result.any()
    result=oracle_region_rejection(world,local,torch.tensor([0]),torch.tensor([False]),2.,1,diagnostic=True)
    assert not result.any()


def test_matching_region_is_retained():
    result=oracle_region_rejection(torch.zeros(2,2),torch.zeros(1,2),torch.tensor([1]),torch.tensor([True]),2.,1,diagnostic=True)
    assert not result.any()
