import torch
from graphnav.pgvb_local_training import local_pgvb_loss


def test_positive_only_loss_never_uses_unknown_as_negative():
    score=torch.tensor([[1.,2.,100.]],requires_grad=True)
    result=local_pgvb_loss(score,torch.tensor([[1,1,-1]]),[dict(frame_id=0,type='Building')])
    result['total'].backward()
    assert torch.isfinite(result['total']) and score.grad[0,2]==0
    assert result['positive_only_rows']==1 and result['pairs']==0
    assert score.grad[0,:2].abs().sum()>0


def test_same_image_other_positive_in_current_unknown_area_is_not_hard_negative():
    s=torch.randn(2,3,requires_grad=True);labels=torch.tensor([[1,-1,-1],[-1,1,-1]])
    loss=local_pgvb_loss(s,labels,[dict(frame_id=0,type='Road')]*2)
    assert loss['pairs']==0


def test_mixed_legacy_loss_is_unchanged_when_both_regions_are_known_negatives():
    from graphnav.pgvb_training import pgvb_loss
    s=torch.tensor([[2.,-1.,10.],[-1.,2.,-8.]],requires_grad=True)
    labels=torch.tensor([[1,0,-1],[0,1,-1]])
    rows=[dict(frame_id=0,type='Building')]*2
    new=local_pgvb_loss(s,labels,rows);old=pgvb_loss(s,labels,rows)
    for key in ('region','bce','pair','total'):assert torch.allclose(new[key],old[key])
