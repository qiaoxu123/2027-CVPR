import torch
from graphnav.visual_only_diagnostic import visual_only_field


def test_original_visual_residual_on_uniform_prior_has_no_a3_context():
    scores=torch.tensor([[1.,3.,2.,0.]])
    observed=torch.tensor([[True,True,False,False]])
    grid=torch.tensor([[[0.,0.],[1.,0.],[0.,1.],[1.,1.]]])
    gain=torch.tensor(.3)
    field=visual_only_field(scores,observed,grid,hidden=8,top_k=2,visual_gain=gain,residual_bound=.5)
    expected=.5*gain.tanh()*((scores-2)/5).tanh()*observed
    assert torch.equal(field.belief_logits,expected)
    assert not field.candidate_context.any()
    assert field.candidate_ids[0,0]==1
    assert torch.equal(field.belief_logits[~observed],torch.zeros(2))


def test_unobserved_frame_remains_uniform_not_false_absence():
    scores=torch.randn(1,4);valid=torch.zeros(1,4,dtype=torch.bool);uv=torch.rand(1,4,2)
    field=visual_only_field(scores,valid,uv,hidden=8,top_k=2,visual_gain=torch.tensor(.3),residual_bound=.5)
    assert not field.belief_logits.any()
    assert not field.dense_context.any()
