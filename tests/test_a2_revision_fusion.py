import pytest
import torch
from experiments.a2_waldo_landmark_belief.revision_fusion import (
    a0_candidate_context,proposal_distribution,bounded_belief_mix,
)


def spatial():
    return dict(base_logits=torch.tensor([.4,.3,.2,.1]).log(),
                grid_world=torch.tensor([[0.,0.],[10.,0.],[20.,0.],[30.,0.]]),
                candidate_xy=torch.tensor([[30.,0.],[0.,0.]]),
                candidate_support=torch.tensor([[False,False,False,True],[True,False,False,False]]))


def test_global_bounded_mass_can_correct_a_peak_single_cell_local_mix_cannot():
    s=spatial();q=torch.tensor([0.,0.,0.,1.]);base=s['base_logits']
    mixed=bounded_belief_mix(base,q,torch.tensor(.25))
    assert int(mixed.argmax())==3 and int(base.argmax())==0
    assert torch.allclose(mixed.softmax(-1),.75*base.softmax(-1)+.25*q,atol=1e-7)
    assert torch.allclose(mixed.logsumexp(0),base.logsumexp(0),atol=1e-7)


def test_off_and_missing_proposal_return_original_object_exactly():
    s=spatial();base=s['base_logits']
    assert bounded_belief_mix(base,torch.ones(4)/4,0.) is base
    assert bounded_belief_mix(base,torch.zeros(4),.25) is base
    with pytest.raises(ValueError):bounded_belief_mix(base,torch.ones(4)/4,.3)


def test_proposal_has_no_mass_outside_public_support_and_has_finite_gradients():
    s=spatial();scores=torch.tensor([1.,0.],requires_grad=True);trust=torch.tensor([.5,.2],requires_grad=True)
    q,valid=proposal_distribution(s,scores,trust,torch.tensor([True,True]))
    assert valid and q[1:3].sum()==0 and torch.isclose(q.sum(),torch.tensor(1.))
    gate=torch.tensor(.1,requires_grad=True);mixed=bounded_belief_mix(s['base_logits'],q,gate)
    loss=-mixed.log_softmax(-1)[3];loss.backward()
    for v in (scores,trust,gate):assert v.grad is not None and torch.isfinite(v.grad).all() and v.grad.abs().sum()>0


def test_empty_reference_or_empty_support_yields_zero_proposal():
    s=spatial();q,valid=proposal_distribution(s,torch.zeros(2),torch.zeros(2),torch.zeros(2,dtype=torch.bool))
    assert not valid and q.sum()==0
    s['candidate_support'].zero_();q,valid=proposal_distribution(s,torch.zeros(2),torch.zeros(2),torch.ones(2,dtype=torch.bool))
    assert not valid and q.sum()==0


def test_a0_context_uses_current_belief_not_private_targets():
    s=spatial();a=a0_candidate_context(s)
    assert a.shape==(2,6) and torch.isfinite(a).all()
    s['base_logits']=s['base_logits'].flip(0);b=a0_candidate_context(s)
    assert not torch.equal(a,b)
    s['candidate_support'][0].zero_();assert torch.isfinite(a0_candidate_context(s)).all()


def test_batched_fusion_matches_single_views_with_empty_rows_and_finite_gradients():
    from experiments.a2_waldo_landmark_belief.revision_fusion import batch_proposal,batch_belief_mix
    s=spatial();b={k:torch.stack((v,v)) for k,v in s.items()};b['candidate_support'][1].zero_()
    scores=torch.tensor([[1.,0.],[0.,1.]],requires_grad=True);trust=torch.zeros(2,2,requires_grad=True)
    q,valid=batch_proposal(b,scores,trust,torch.ones(2,2,dtype=torch.bool))
    expected,_=proposal_distribution(s,scores[0],trust[0],torch.ones(2,dtype=torch.bool))
    assert torch.allclose(q[0],expected) and valid.tolist()==[True,False] and q[1].sum()==0
    gate=torch.tensor([.1,.1],requires_grad=True);mixed=batch_belief_mix(b['base_logits'],q,gate)
    assert torch.allclose(mixed[0],bounded_belief_mix(s['base_logits'],expected,gate[0]))
    assert torch.equal(mixed[1],b['base_logits'][1])
    (-mixed.log_softmax(-1)[:,3].mean()).backward()
    for value in (scores,trust,gate):assert torch.isfinite(value.grad).all()
