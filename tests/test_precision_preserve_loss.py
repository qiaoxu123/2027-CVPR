import torch
from graphnav.precision_preserve_loss import preservation_kl


def test_preservation_only_uses_correct_prior_rows_and_pushes_back_to_prior():
    prior=torch.tensor([[3.,0.,0.,0.],[3.,0.,0.,0.]])
    updated=torch.tensor([[0.,3.,0.,0.],[0.,3.,0.,0.]],requires_grad=True)
    world=torch.tensor([[[0.,0.],[40.,0.],[80.,0.],[120.,0.]]]).repeat(2,1,1)
    target=torch.tensor([[0.,0.],[120.,0.]])
    loss,count=preservation_kl(prior,updated,world,target)
    assert count==1 and loss>0
    loss.backward();assert updated.grad[0,0]<0 and updated.grad[0,1]>0
    assert updated.grad[1].abs().sum()==0


def test_prior_equal_posterior_has_zero_preservation_loss():
    prior=torch.randn(2,4);world=torch.zeros(2,4,2);target=torch.zeros(2,2)
    loss,count=preservation_kl(prior,prior,world,target)
    assert count==2
    torch.testing.assert_close(loss,torch.tensor(0.),atol=1e-6,rtol=0)
