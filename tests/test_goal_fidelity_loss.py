import torch
from graphnav.goal_hit_training import fidelity_hit_loss


def inputs(target):
 return torch.tensor([[[0.,0.],[.5,.5],[1.,1.]]]),torch.tensor([[0.,0.,100.,100.]]),torch.tensor([target])

def test_correct_source_is_stationary_at_identical_logits():
 z=torch.tensor([[3.,0.,0.]],requires_grad=True);g,b,t=inputs([0.,0.]);loss,p=fidelity_hit_loss(z,z.detach().clone(),g,b,t);loss.backward()
 assert p['source_correct']==1 and z.grad.abs().max()<1e-6

def test_wrong_source_receives_real_correction_gradient():
 z=torch.tensor([[0.,3.,0.]],requires_grad=True);g,b,t=inputs([0.,0.]);loss,p=fidelity_hit_loss(z,z.detach().clone(),g,b,t);loss.backward()
 assert p['source_wrong']==1 and z.grad[0,0]<0 and z.grad[0,1]>0

def test_uncovered_target_does_not_produce_nan():
 z=torch.zeros(1,3,requires_grad=True);g,b,t=inputs([400.,400.]);loss,p=fidelity_hit_loss(z,z.detach().clone(),g,b,t);loss.backward()
 assert p['no_positive20']==1 and torch.isfinite(loss) and torch.isfinite(z.grad).all()
