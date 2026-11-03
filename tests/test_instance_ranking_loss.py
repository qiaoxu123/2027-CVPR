import torch
from graphnav.instance_ranking_loss import instance_ranking_loss


def test_target_is_ranked_above_other_instance_and_unknown_has_no_gradient():
    score=torch.tensor([[0.,1.,100.]],requires_grad=True)
    labels=torch.tensor([[1,0,-1]],dtype=torch.int8)
    loss,counts=instance_ranking_loss(score,labels,torch.ones_like(labels,dtype=torch.bool))
    assert counts['rows']==1
    loss.backward();assert score.grad[0,0]<0 and score.grad[0,1]>0 and score.grad[0,2]==0
    same,_=instance_ranking_loss(torch.tensor([[0.,1.,-100.]]),labels,torch.ones_like(labels,dtype=torch.bool))
    torch.testing.assert_close(loss,same)


def test_absent_or_invalid_target_does_not_create_negative_evidence():
    score=torch.randn(2,3,requires_grad=True);labels=torch.tensor([[0,0,-1],[1,0,-1]])
    valid=torch.tensor([[True,True,True],[False,True,True]])
    loss,counts=instance_ranking_loss(score,labels,valid)
    assert counts['rows']==0 and loss==0
    loss.backward();assert score.grad.abs().sum()==0


def test_noncar_position_auxiliary_does_not_assign_car_background_positive():
    from graphnav.instance_ranking_loss import noncar_position_auxiliary
    scores=torch.tensor([[0.,0.],[0.,0.]],requires_grad=True)
    distance=torch.tensor([[1.,10.],[1.,10.]])
    loss=noncar_position_auxiliary(scores,distance,torch.ones(2,2,dtype=torch.bool),['Car','Building'])
    loss.backward();assert scores.grad[0].abs().sum()==0 and scores.grad[1].abs().sum()>0
