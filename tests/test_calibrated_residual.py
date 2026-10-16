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
