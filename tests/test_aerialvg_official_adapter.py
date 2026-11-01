"""Integration tests against downloaded official source and checkpoint, not a mock."""
import torch
from test_calibrated_residual import fixture
from graphnav.calibrated_visual_residual import CalibratedVisualResidual


def test_official_pretrained_adapter_identity_gradients_and_unknown_mask():
    torch.manual_seed(7);model=CalibratedVisualResidual(official_relation=True)
    base,public,inputs=fixture();tokens=torch.randn(2,12,768);mask=torch.ones(2,12,dtype=torch.bool);mask[:,-2:]=False
    delta,_,_=model(base,public,inputs,tokens,mask)
    assert not delta.any()
    with torch.no_grad():model.official_relation.gain.fill_(.2)
    delta,_,_=model(base,public,inputs,tokens,mask)
    assert torch.isfinite(delta).all() and delta.abs().max()<=model.bound
    changed=model(base,public,inputs,tokens+torch.randn_like(tokens),mask)[0]
    assert not torch.allclose(delta,changed)
    delta.square().sum().backward()
    assert model.official_relation.visual[0].weight.grad.abs().sum()>0
    assert model.official_relation.language[0].weight.grad.abs().sum()>0
    assert all(p.grad is None and not p.requires_grad for p in model.official_relation.pairs.parameters())
    public['roi_features'].zero_();public['history_features'].zero_()
    assert not model(base,public,inputs,tokens,mask)[0].any()


def test_official_adapter_preserves_shared_initialization():
    torch.manual_seed(17);plain=CalibratedVisualResidual()
    torch.manual_seed(17);official=CalibratedVisualResidual(official_relation=True)
    for name,value in plain.state_dict().items():assert torch.equal(value,official.state_dict()[name]),name
