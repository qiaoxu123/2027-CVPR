import torch
from graphnav.world_likelihood_evidence import world_evidence_logits


def test_zero_evidence_is_neutral_for_nonuniform_prior_and_dense_local_sampling():
    prior=torch.tensor([[3.,2.,1.,0.,-1.,2.,1.,0.,3.,2.,1.,0.,-1.,2.,1.,0.]])
    uv=torch.full((1,144,2),.6);valid=torch.ones(1,144,dtype=torch.bool)
    result=world_evidence_logits(prior,uv,torch.zeros(1,144),valid)
    torch.testing.assert_close(result.log_softmax(-1),prior.log_softmax(-1))


def test_duplicate_sampling_does_not_multiply_evidence_and_outside_points_are_ignored():
    prior=torch.randn(1,16);uv=torch.tensor([[[.4,.4],[.6,.6]]]);scores=torch.tensor([[2.,-1.]],requires_grad=True);valid=torch.ones(1,2,dtype=torch.bool)
    result=world_evidence_logits(prior,uv,scores,valid)
    repeat=world_evidence_logits(prior,uv.repeat_interleave(3,1),scores.repeat_interleave(3,1),valid.repeat_interleave(3,1))
    torch.testing.assert_close(result,repeat)
    extra=world_evidence_logits(prior,torch.cat([uv,torch.tensor([[[2.,2.]]])],1),torch.cat([scores,torch.tensor([[100.]])],1),torch.ones(1,3,dtype=torch.bool))
    torch.testing.assert_close(result,extra)
    result.sum().backward();assert scores.grad.abs().sum()>0


def test_v3_has_finite_visual_gradients_and_no_observation_preserves_prior():
    from graphnav.visual_chain_position import WorldLikelihoodPositionHead,visual_chain_loss
    from test_local_chain_position import inputs
    x=inputs();model=WorldLikelihoodPositionHead(dimension=12,hidden=8)
    joint,visual=model(**x,return_visual=True)
    world=torch.rand(2,21,2)*40
    loss=visual_chain_loss(model,joint,visual,x,world,world[:,0],['Building','Car'],['building','white car'])['total']
    loss.backward()
    assert model.image.weight.grad.norm()>0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    x['local_valid'].fill_(False)
    output=model(**x)
    torch.testing.assert_close(output[:,:16].log_softmax(-1),x['prior_logits'].log_softmax(-1))
