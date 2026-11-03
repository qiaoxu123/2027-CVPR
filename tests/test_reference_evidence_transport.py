import inspect
import torch
from graphnav.reference_evidence_transport import ReferenceEvidenceTransport


def inputs():
    torch.manual_seed(4)
    return dict(global_world=torch.tensor([[[0.,0.],[50.,0.],[150.,0.]]]),local_world=torch.tensor([[[0.,0.],[10.,10.]]]),
        image_features=torch.randn(1,2,3,8),reference_text=torch.randn(1,8),instruction_text=torch.randn(1,8),local_valid=torch.tensor([[True,True]]))


def test_zero_initialization_and_no_observation_are_exactly_neutral():
    model=ReferenceEvidenceTransport(dimension=8,hidden=4);x=inputs()
    assert torch.equal(model(**x),torch.zeros(1,3))
    model.evidence.weight.data.fill_(1.)
    assert torch.equal(model(**dict(x,local_valid=torch.zeros_like(x['local_valid']))),torch.zeros(1,3))
    assert torch.equal(model(**dict(x,image_features=torch.zeros_like(x['image_features']))),torch.zeros(1,3))
    assert torch.equal(model(**dict(x,reference_text=torch.zeros_like(x['reference_text']))),torch.zeros(1,3))


def test_real_evidence_can_change_global_cells_outside_the_observed_positions():
    model=ReferenceEvidenceTransport(dimension=8,hidden=4);model.evidence.weight.data.fill_(1.);x=inputs()
    assert model(**x)[0,-1].abs()>0
    assert not {'target_world','target_id','labels','contour'}&set(inspect.signature(model.forward).parameters)


def test_relative_geometry_translation_and_duplicate_sampling_invariance():
    model=ReferenceEvidenceTransport(dimension=8,hidden=4);model.evidence.weight.data.fill_(1.);x=inputs();old=model(**x)
    shifted=dict(x,global_world=x['global_world']+1000,local_world=x['local_world']+1000)
    assert torch.allclose(old,model(**shifted),atol=1e-6)
    duplicated=dict(x,local_world=x['local_world'].repeat_interleave(2,1),image_features=x['image_features'].repeat_interleave(2,1),local_valid=x['local_valid'].repeat_interleave(2,1))
    assert torch.allclose(old,model(**duplicated),atol=1e-6)


def test_image_gradient_is_effective_and_all_invalid_backward_is_finite():
    model=ReferenceEvidenceTransport(dimension=8,hidden=4);model.evidence.weight.data.fill_(1.);x=inputs()
    loss=-model(**x).log_softmax(-1)[0,0];loss.backward()
    assert model.image.weight.grad.norm()>0
    model.zero_grad();model(**dict(x,local_valid=torch.zeros_like(x['local_valid']))).sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def wrapper_inputs():
    x=inputs();x['global_world']=torch.tensor([[[0.,0.],[50.,0.],[0.,50.],[50.,50.]]]);x['local_world']=torch.zeros(1,2,2)
    return dict(prior_logits=torch.randn(1,4),local_uv=torch.zeros(1,2,2),image_features=x['image_features'],target_text=torch.randn(1,8),context_text=x['reference_text'],
        public_geometry=torch.randn(1,2,31),quality=torch.ones(1,2,6),local_valid=x['local_valid'],global_world=x['global_world'],local_world=x['local_world'],instruction_text=x['instruction_text'])


def test_wrapper_preserves_frozen_base_and_new_module_only_receives_gradients():
    from graphnav.visual_chain_position import WorldLikelihoodPositionHead
    from graphnav.reference_evidence_transport import ReferenceTransportPositionHead
    model=ReferenceTransportPositionHead(WorldLikelihoodPositionHead(dimension=8,hidden=4),dimension=8,hidden=4);x=wrapper_inputs()
    base=model.base(**{k:v for k,v in x.items() if k not in ('global_world','local_world','instruction_text')})
    assert torch.equal(model(**x),base)
    model.transport.evidence.weight.data.fill_(1.)
    pred,evidence=model(**x,return_evidence=True);assert evidence[0,3].abs()>0
    (-pred[:,:4].log_softmax(-1)[0,3]).backward()
    assert all(p.grad is None for p in model.base.parameters())
    assert model.transport.image.weight.grad.norm()>0
    assert torch.equal(model(**x,control='zero-image'),base)


def test_local_control_does_not_update_outside_coverage_and_constant_has_no_current_rgb():
    from graphnav.visual_chain_position import WorldLikelihoodPositionHead
    from graphnav.reference_evidence_transport import ReferenceTransportPositionHead
    x=wrapper_inputs();model=ReferenceTransportPositionHead(WorldLikelihoodPositionHead(dimension=8,hidden=4),mode='local',dimension=8,hidden=4)
    model.transport.evidence.weight.data.fill_(1.)
    _,energy=model(**x,return_evidence=True)
    assert torch.equal(energy[0,1:],torch.zeros(3))
    model.mode='constant';model.constant_features.copy_(torch.randn(3,8))
    _,old=model(**x,return_evidence=True)
    _,new=model(**dict(x,image_features=torch.randn_like(x['image_features'])),return_evidence=True)
    assert torch.equal(old,new)
