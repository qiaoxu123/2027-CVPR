import pytest
import torch
from graphnav.pgvb_training import pgvb_loss


def test_weak_boundary_is_not_forced_positive_or_negative_and_pairs_are_same_image():
    scores=torch.tensor([[2.,-1.,100.],[-1.,2.,-100.]],requires_grad=True)
    labels=torch.tensor([[1,0,-1],[0,1,-1]])
    rows=[dict(frame_id=3,type='Building'),dict(frame_id=3,type='Building')]
    loss=pgvb_loss(scores,labels,rows);loss['total'].backward()
    assert loss['pairs']==2 and torch.isfinite(loss['total'])
    assert (scores.grad[:,2]==0).all()
    rows[1]['frame_id']=4
    assert pgvb_loss(scores,labels,rows)['pairs']==0


def test_ambiguous_or_unobserved_region_is_rejected_as_supervision():
    with pytest.raises(ValueError,match='Ambiguous'):
        pgvb_loss(torch.randn(1,3),torch.tensor([[-1,0,0]]),[dict(frame_id=0,type='Road')])
