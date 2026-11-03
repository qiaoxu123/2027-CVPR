import torch
from torch import nn
from graphnav.public_geometry_visual import PublicGeometryVisualHead


class Base(nn.Module):
    def forward(self,**x):
        return torch.cat((x['prior_logits'],torch.full_like(x['local_valid'],float('-inf'),dtype=torch.float32)),-1)


def inputs():
    torch.manual_seed(0)
    return dict(prior_logits=torch.randn(2,16),local_uv=torch.rand(2,4,2),image_features=torch.randn(2,4,3,8),target_text=torch.randn(2,8),context_text=torch.randn(2,8),public_geometry=torch.zeros(2,4,31),quality=torch.ones(2,4,6),local_valid=torch.ones(2,4,dtype=torch.bool),world_geometry=torch.randn(2,20,33),instruction_text=torch.randn(2,8))


def test_zero_initialization_preserves_same_call_base_and_no_image_is_neutral():
    model=PublicGeometryVisualHead(Base(),dimension=8,hidden=4);x=inputs()
    out,base,m,v=model(**x,return_base=True)
    assert torch.equal(out,base) and torch.equal(m,torch.zeros_like(m)) and torch.equal(v,torch.zeros_like(v))
    with torch.no_grad():model.visual_score.weight.fill_(.5)
    _,_,_,v=model(**x,control='zero',return_base=True)
    assert torch.equal(v,torch.zeros_like(v))


def test_visual_gradients_flow_after_output_initialization_and_map_is_image_independent():
    x=inputs();model=PublicGeometryVisualHead(Base(),dimension=8,hidden=4)
    with torch.no_grad():model.visual_score.weight.fill_(.1)
    out=model(**x);out[:,:16].square().mean().backward()
    assert model.image.weight.grad.abs().sum()>0
    model=PublicGeometryVisualHead(Base(),mode='map',dimension=8,hidden=4)
    a=model(**x);x['image_features']*=0
    assert torch.equal(a,model(**x))
