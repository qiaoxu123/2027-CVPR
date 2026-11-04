import numpy as np
import torch
from graphnav.polygon_guided_visual import PolygonGuidedVisualBelief,polygon_shape_tokens


def test_polygon_shape_does_not_copy_absolute_world_position():
    polygon=np.array([[10,10],[20,10],[20,25],[10,25]],float)
    footprint=np.array([[0,0],[100,0],[100,100],[0,100]],float)
    first=polygon_shape_tokens([polygon],footprint)
    other=polygon_shape_tokens([polygon+30],footprint)
    assert np.allclose(first,other,atol=1e-5)
    assert not np.allclose(first,polygon_shape_tokens([polygon*2],footprint))


def test_cross_attention_gradients_and_invalid_image_tokens():
    torch.manual_seed(4);model=PolygonGuidedVisualBelief(dimension=8,hidden=16,heads=2)
    image=torch.randn(2,6,3,8,requires_grad=True);text=torch.randn(2,8);poly=torch.randn(2,16,4)
    valid=torch.tensor([[1,1,1,0,0,0],[0,0,0,0,0,0]],dtype=torch.bool)
    score=model(image_features=image,name_text=text,type_text=text,polygon_tokens=poly,valid=valid)
    assert torch.isfinite(score).all() and (score[1]==0).all()
    score[0,:3].square().mean().backward()
    assert image.grad[0,:3].abs().sum()>0 and image.grad[0,3:].abs().sum()==0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.attention.in_proj_weight.grad.abs().sum()>0


def test_zero_rgb_cannot_select_using_absolute_polygon_location():
    model=PolygonGuidedVisualBelief(dimension=8,hidden=16,heads=2)
    z=torch.zeros(1,8,3,8);t=torch.randn(1,8);valid=torch.ones(1,8,dtype=torch.bool)
    scores=model(image_features=z,name_text=t,type_text=t,polygon_tokens=torch.randn(1,16,4),valid=valid)
    assert torch.allclose(scores,scores[:,:1].expand_as(scores),atol=1e-6)
