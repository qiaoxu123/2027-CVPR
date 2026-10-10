import torch
from graphnav.static_controls import graph_control, allowed_region_loss
from graphnav.semantic_geometry import EDGE_TYPES


def test_controls_preserve_source_and_types():
    edges = torch.ones(1,3,3,8+len(EDGE_TYPES))
    x = {'edges':edges,'adjacency':torch.ones(1,3,3,dtype=torch.bool)}
    z = graph_control(x,'zero-geometry')
    assert not z['edges'][...,:8].any()
    assert torch.equal(z['edges'][...,8:],edges[...,8:])
    assert edges.all()
    assert graph_control(x,'self')['adjacency'].sum()==3


def test_legal_alternatives_are_not_negatives():
    logits=torch.tensor([[0.,0.,0.]],requires_grad=True)
    allowed=torch.tensor([[True,True,False]])
    loss=allowed_region_loss(logits,allowed,torch.tensor([True]))
    loss.backward()
    assert logits.grad[0,0]<0 and logits.grad[0,1]<0 and logits.grad[0,2]>0


def test_no_eligible_relations_returns_differentiable_zero():
    x=torch.randn(2,3,requires_grad=True)
    loss=allowed_region_loss(x,torch.ones_like(x,dtype=torch.bool),torch.ones(2,dtype=torch.bool))
    loss.backward()
    assert loss==0 and not x.grad.any()
