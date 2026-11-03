import pytest
import torch
from graphnav.evidence_reachability import local_evidence_coverage,position_support_bound
from graphnav.world_likelihood_evidence import world_evidence_logits


def test_uncovered_prior_order_cannot_be_changed_by_local_rgb_scores():
    uv=torch.tensor([[[0.,0.]]]);valid=torch.tensor([[True]])
    coverage=local_evidence_coverage(uv,valid,4)[0]
    assert coverage.tolist()==[True,False,False,False]
    prior=torch.tensor([3.,2.,1.,0.]);success=torch.tensor([False,False,True,False])
    assert position_support_bound(prior,coverage,success,diagnostic=True)['possible'] is False
    for score in [-10000.,0.,10000.]:
        updated=world_evidence_logits(prior[None],uv,torch.tensor([[score]]),valid)[0]
        assert updated.argmax().item()!=2
        assert torch.allclose(updated[1:]-updated[1],prior[1:]-prior[1])


def test_outside_prior_can_be_recovered_only_if_its_unchanged_best_is_correct():
    prior=torch.tensor([3.,2.,1.,0.]);coverage=torch.tensor([True,False,False,False]);success=torch.tensor([False,True,False,False])
    result=position_support_bound(prior,coverage,success,diagnostic=True)
    assert result['possible'] and result['kind']=='unchanged-prior-best-correct'
    with pytest.raises(ValueError):position_support_bound(prior,coverage,success)


def test_covered_success_is_only_optimistic_because_splat_cells_can_be_coupled():
    result=position_support_bound(torch.tensor([3.,2.,1.,0.]),torch.tensor([True,True,False,False]),torch.tensor([False,True,False,False]),diagnostic=True)
    assert result['possible'] and result['kind']=='covered-success-cell-optimistic'
