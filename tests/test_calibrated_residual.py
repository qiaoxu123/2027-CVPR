from types import SimpleNamespace
import torch
from graphnav.calibrated_visual_residual import CalibratedVisualResidual,correction_loss


def fixture():
    batch,cells,dim=2,9,1024
    base=SimpleNamespace(static=SimpleNamespace(belief_logits=torch.randn(batch,cells),dense_context=torch.randn(batch,cells,128)),current_scores=torch.randn(batch,cells),appearance_scores=torch.randn(batch,cells))
    public=dict(roi_features=torch.randn(batch,cells,3,dim),roi_valid=torch.ones(batch,cells,3,dtype=torch.bool),roi_quality=torch.ones(batch,cells,3,3),history_features=torch.randn(batch,cells,3,dim),history_valid=torch.ones(batch,cells,3,dtype=torch.bool),history_quality=torch.ones(batch,cells,3,3),native_text=torch.randn(batch,dim),appearance_text=torch.randn(batch,dim),pose_world=torch.zeros(batch,4),bounds=torch.tensor([[0.,0.,100.,100.]]).repeat(batch,1))
    inputs=dict(grid_xy=torch.rand(batch,cells,2))
    return base,public,inputs


def test_zero_initialization_and_unknown_evidence_are_identity():
    model=CalibratedVisualResidual();base,public,inputs=fixture()
    delta,_,_=model(base,public,inputs);assert not delta.any()
    with torch.no_grad():model.network[-1].weight.normal_()
    public['roi_features'].zero_();public['history_features'].zero_()
    delta,_,known=model(base,public,inputs)
    assert not delta.any();assert not known.any()


def test_bounded_image_dependent_correction_and_legal_region_loss():
    model=CalibratedVisualResidual();base,public,inputs=fixture()
    with torch.no_grad():model.network[-1].weight.normal_()
    delta,_,known=model(base,public,inputs)
    assert delta.abs().max()<=2
    second={**public,'roi_features':public['roi_features'].flip(1),'history_features':public['history_features'].flip(1)}
    changed,_,_=model(base,second,inputs);assert not torch.equal(delta,changed)
    world=inputs['grid_xy']*100
    loss=correction_loss(base.static.belief_logits,delta,world,torch.zeros(2,3),known)
    loss['total'].backward()
    assert torch.isfinite(loss['total'])
    assert model.network[-1].weight.grad.abs().sum()>0


def test_absolute_evidence_can_lower_all_observed_cells_without_penalizing_unknown():
    model=CalibratedVisualResidual(centered=False);base,public,inputs=fixture()
    with torch.no_grad():model.network[-1].bias[0]=-1.
    public['roi_valid'][:,0]=False;public['history_valid'][:,0]=False
    delta,_,known=model(base,public,inputs)
    assert not delta[:,0].any()
    assert (delta[:,1:]<0).all()


def test_relation_anchors_ignore_invalid_features_and_receive_gradients():
    torch.manual_seed(17)
    model=CalibratedVisualResidual(relation=True);base,public,inputs=fixture()
    assert not model(base,public,inputs)[0].any()
    with torch.no_grad():model.network[-1].weight.normal_()
    public['roi_valid'][:,0]=False;public['history_valid'][:,0]=False
    delta,_,known=model(base,public,inputs)
    altered={**public,'roi_features':public['roi_features'].clone(),'history_features':public['history_features'].clone()}
    altered['roi_features'][:,0]=1000;altered['history_features'][:,0]=-1000
    assert torch.equal(delta,model(base,altered,inputs)[0])
    assert not delta[:,0].any()
    correction_loss(base.static.belief_logits,delta,inputs['grid_xy']*100,torch.zeros(2,3),known)['total'].backward()
    assert model.relation[0].weight.grad.abs().sum()>0
    public['roi_quality'].zero_();public['history_quality'].zero_()
    assert not model(base,public,inputs)[0].any()


def test_relation_checkpoint_roundtrip():
    model=CalibratedVisualResidual(relation=True)
    restored=CalibratedVisualResidual(relation=True)
    restored.load_state_dict(model.state_dict(),strict=True)
    base,public,inputs=fixture()
    assert torch.equal(model(base,public,inputs)[0],restored(base,public,inputs)[0])


def test_relation_transmits_other_observed_candidate_visual_evidence():
    torch.manual_seed(23)
    model=CalibratedVisualResidual(relation=True,centered=False)
    base,public,inputs=fixture()
    with torch.no_grad():model.network[-1].weight.normal_()
    before=model(base,public,inputs)[0][:,1].clone()
    public['roi_features'][:,0]*=-1
    after=model(base,public,inputs)[0][:,1]
    assert not torch.allclose(before,after)
    public['roi_valid'].zero_();public['history_valid'].zero_()
    delta,confidence,known=model(base,public,inputs)
    assert not delta.any() and not known.any() and torch.isfinite(confidence).all()


def test_attention_relation_uses_named_geometry_and_backpropagates():
    torch.manual_seed(31)
    model=CalibratedVisualResidual(relation=True,relation_attention=True)
    base,public,inputs=fixture()
    inputs['node_valid']=torch.ones(2,3,dtype=torch.bool)
    inputs['pair_geometry']=torch.rand(2,9,3,10)
    tokens=torch.randn(2,7,768);mask=torch.ones(2,7,dtype=torch.bool);nodes=torch.randn(2,3,768)
    assert not model(base,public,inputs,tokens,mask,nodes)[0].any()
    with torch.no_grad():model.network[-1].weight.normal_()
    delta,_,_=model(base,public,inputs,tokens,mask,nodes)
    delta.square().sum().backward()
    assert model.reference_key.weight.grad.abs().sum()>0
    assert model.relation_attention[0].weight.grad.abs().sum()>0
    public['roi_valid'].zero_();public['history_valid'].zero_()
    assert not model(base,public,inputs,tokens,mask,nodes)[0].any()


def test_hard_ranking_skips_unobserved_targets_and_pushes_correct_direction():
    from graphnav.calibrated_visual_residual import observed_hard_ranking
    logits=torch.zeros(1,3);delta=torch.zeros(1,3,requires_grad=True)
    world=torch.tensor([[[0.,0.],[30.,0.],[60.,0.]]]);target=torch.zeros(1,3)
    visual=torch.ones(1,3,4);known=torch.ones(1,3,dtype=torch.bool)
    loss=observed_hard_ranking(logits,delta,world,target,known,visual)
    loss.backward()
    assert delta.grad[0,0]<0 and delta.grad[0,1:].sum()>0
    known[:,0]=False
    assert observed_hard_ranking(logits,delta,world,target,known,visual)==0


def test_optional_relation_preserves_shared_initial_weights():
    torch.manual_seed(0);baseline=CalibratedVisualResidual()
    torch.manual_seed(0);relation=CalibratedVisualResidual(relation=True,relation_attention=True)
    for name,value in baseline.state_dict().items():
        assert torch.equal(value,relation.state_dict()[name]),name
