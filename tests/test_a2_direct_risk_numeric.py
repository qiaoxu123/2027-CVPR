import pytest
import torch
from experiments.a2_waldo_landmark_belief.direct_risk_numeric import audit_gpu_proposal
from experiments.a2_waldo_landmark_belief.revision_fusion import proposal_distribution


def test_world_coordinate_fp32_distance_roundoff_is_not_false_corruption():
    torch.manual_seed(3)
    xy=torch.tensor([[1234.2,1783.4],[1240.7,1794.1]])
    grid=torch.randn(784,2)*15+torch.tensor([1234.,1783.])
    spatial=dict(base_logits=torch.zeros(784),grid_world=grid,candidate_xy=xy,candidate_support=torch.ones(2,784,dtype=torch.bool))
    prediction=dict(scores=torch.tensor([.2,-.1]),candidate_match_logits=torch.tensor([.5,.7]),reliable=torch.ones(2,dtype=torch.bool))
    q,available=proposal_distribution(spatial,prediction['scores'],prediction['candidate_match_logits'],prediction['reliable'])
    report=audit_gpu_proposal(spatial,prediction,q,available)
    assert report['passed'] and torch.isfinite(q).all()
    bad=torch.zeros_like(q);bad[-1]=1.
    with pytest.raises(ValueError):audit_gpu_proposal(spatial,prediction,bad,available)


def test_probability_outside_public_reliable_support_is_always_rejected():
    spatial=dict(base_logits=torch.zeros(2),grid_world=torch.tensor([[0.,0.],[1.,0.]]),candidate_xy=torch.zeros(1,2),candidate_support=torch.tensor([[True,False]]))
    prediction=dict(scores=torch.zeros(1),candidate_match_logits=torch.zeros(1),reliable=torch.ones(1,dtype=torch.bool))
    with pytest.raises(ValueError):audit_gpu_proposal(spatial,prediction,torch.tensor([.9,.1]),True)
