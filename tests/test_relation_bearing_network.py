import torch


def inputs():
    torch.manual_seed(42);b,r,n,hops=2,2,6,4
    return dict(tokens=torch.randn(b,5,8),token_mask=torch.ones(b,5,dtype=torch.bool),node_names=torch.randn(b,n,8),node_shape=torch.randn(b,n,39),node_types=torch.zeros(b,n,dtype=torch.long),node_valid=torch.ones(b,n,dtype=torch.bool),edges=torch.randn(b,n,n,10),anchor_indices=torch.tensor([[0,1],[1,2]]),anchor_valid=torch.tensor([[True,True],[True,False]]),entity_steps=torch.randn(b,r,hops,8),relation_steps=torch.randn(b,r,hops,8),step_valid=torch.tensor([[[True,True,False,False],[True,False,False,False]],[[True,False,False,False],[True,False,False,False]]]),angle_kernel=torch.rand(b,r,n,8).softmax(-1))


def test_joint_probabilities_and_actual_gradients():
    from graphnav.relation_bearing_network import RolePolygonBearing
    model=RolePolygonBearing(text_dim=8,shape_dim=39,hidden=16)
    x=inputs();q=model(**x,use_relations=True)
    assert q['probabilities'].shape==(2,2,8)
    torch.testing.assert_close(q['probabilities'].sum(-1),torch.ones(2,2))
    assert torch.isfinite(q['entropy']).all()
    loss=-q['probabilities'][0,0,2].log()-q['object_probabilities'][0,0,3].log();loss.backward()
    assert model.shape.weight.grad.abs().sum()>0
    assert model.relation.weight.grad.abs().sum()>0
    assert model.name.weight.grad.abs().sum()>0


def test_relational_inputs_have_effect_but_control_does_not_use_them():
    from graphnav.relation_bearing_network import RolePolygonBearing
    model=RolePolygonBearing(text_dim=8,shape_dim=39,hidden=16).eval();x=inputs();other=dict(x,relation_steps=x['relation_steps']+3.)
    assert not torch.allclose(model(**x,use_relations=True)['probabilities'],model(**other,use_relations=True)['probabilities'])
    torch.testing.assert_close(model(**x,use_relations=False)['probabilities'],model(**other,use_relations=False)['probabilities'])
