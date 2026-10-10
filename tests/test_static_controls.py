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


def test_reliable_links_skip_ambiguous_anchors_and_surface_along():
    import numpy as np
    from graphnav.semantic_graph import prepare_semantic_graph
    landmarks=[{'name':'Tower','contour':[[.2,.2],[.3,.2],[.3,.3],[.2,.3]]},
               {'name':'Tower','contour':[[.6,.6],[.7,.6],[.7,.7],[.6,.7]]}]
    graph=prepare_semantic_graph(landmarks, [], 'Find the target east of Tower',
                                 [0,0,100,100], max_roads=0)
    inputs={key:torch.from_numpy(graph[source])[None] for key,source in
            [('adjacency','adjacency'),('edges','edges'),('nodes','nodes'),('node_types','node_types')]}
    original=inputs['adjacency'].clone()
    result=graph_control(inputs,'reliable-relation',[graph['node_names']])
    assert torch.equal(result['adjacency'],torch.diag_embed(original.diagonal(dim1=-2,dim2=-1)))
    assert torch.equal(inputs['adjacency'],original)
    graph=prepare_semantic_graph(landmarks[:1], [], 'Find the target east of Tower',
                                 [0,0,100,100], max_roads=0)
    inputs={key:torch.from_numpy(graph[source])[None] for key,source in
            [('adjacency','adjacency'),('edges','edges'),('nodes','nodes'),('node_types','node_types')]}
    result=graph_control(inputs,'reliable-relation',[graph['node_names']])
    assert result['adjacency'][0,0,1] and result['adjacency'][0,1,0]
