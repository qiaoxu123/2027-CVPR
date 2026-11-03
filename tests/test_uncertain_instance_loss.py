import math
import torch
from graphnav.uncertain_instance_loss import uncertain_instance_loss


def test_soft_preference_has_finite_optimum_not_hard_positive():
    scores=torch.tensor([[math.log(.65/.35),0.]],requires_grad=True)
    loss,count=uncertain_instance_loss(scores,torch.tensor([[1,0]]),torch.ones(1,2,dtype=torch.bool),torch.tensor([.65]))
    loss.backward()
    assert count==1 and scores.grad.abs().max()<1e-6


def test_unknown_review_or_missing_other_instance_abstains():
    scores=torch.zeros(2,3,requires_grad=True)
    labels=torch.tensor([[1,0,-1],[1,-1,-1]])
    loss,count=uncertain_instance_loss(scores,labels,torch.ones(2,3,dtype=torch.bool),torch.tensor([.5,.8]))
    assert count==0 and loss==0


def test_unlabelled_background_receives_no_identity_gradient():
    scores=torch.zeros(1,3,requires_grad=True)
    loss,count=uncertain_instance_loss(scores,torch.tensor([[1,0,-1]]),torch.ones(1,3,dtype=torch.bool),torch.tensor([.65]))
    loss.backward()
    assert count==1 and scores.grad[0,0]<0 and scores.grad[0,1]>0 and scores.grad[0,2]==0


def test_weak_teacher_does_not_penalize_already_strong_correct_order():
    scores=torch.tensor([[3.,0.]],requires_grad=True)
    loss,_=uncertain_instance_loss(scores,torch.tensor([[1,0]]),torch.ones(1,2,dtype=torch.bool),torch.tensor([.65]))
    loss.backward()
    assert loss==0 and not scores.grad.any()
