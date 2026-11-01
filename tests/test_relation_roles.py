import torch
from graphnav.relation_roles import parse_roles
from graphnav.relation_supervision import compass_counterfactual_loss
from graphnav.calibrated_visual_residual import CalibratedVisualResidual
from test_calibrated_residual import fixture


def test_public_roles_handle_typed_names_without_turning_name_words_into_relations():
    r=parse_roles('A red house north of The Gym',['Building The Gym',''])
    assert r['valid'] and r['reference_index']==0 and r['direction']==(0.,1.)
    assert r['target']=='A red house' and r['opposite_text']=='A red house south of The Gym'
    assert not parse_roles('A house near West Road',['TrafficRoad West Road'])['direction']!=(0.,0.)
    assert not parse_roles('A house on West Road',['TrafficRoad West Road'])['valid']
    assert not parse_roles('North of The Gym and south of Nike',['Building The Gym','Building Nike'])['valid']
    assert parse_roles('A car behind St Teresa',['Building St. Teresa'])['relation']=='behind'


def test_structured_roles_zero_init_and_gradients():
    torch.manual_seed(5);m=CalibratedVisualResidual(relation=True,relation_attention=True,structured_relation=True)
    b,p,i=fixture();i['node_valid']=torch.ones(2,3,dtype=torch.bool);i['pair_geometry']=torch.randn(2,9,3,10)
    p.update(role_target=torch.randn(2,768),role_reference=torch.randn(2,768),role_relation=torch.randn(2,768),role_valid=torch.ones(2,dtype=torch.bool),role_direction=torch.tensor([[0.,1.],[1.,0.]]),role_reference_index=torch.tensor([0,1]))
    t=torch.randn(2,7,768);mask=torch.ones(2,7,dtype=torch.bool);names=torch.randn(2,3,768)
    assert not m(b,p,i,t,mask,names)[0].any()
    with torch.no_grad():m.role_score[-1].weight.normal_()
    d=m(b,p,i,t,mask,names)[0];d.square().sum().backward()
    assert m.role_target.weight.grad.abs().sum()>0
    flipped={**p,'role_direction':-p['role_direction']}
    assert not torch.equal(d,m(b,flipped,i,t,mask,names)[0])
    p['roi_valid'].zero_();p['history_valid'].zero_()
    assert not m(b,p,i,t,mask,names)[0].any()


def test_compass_weak_supervision_requires_geometric_agreement():
    real=torch.zeros(1,3,requires_grad=True);opposite=torch.zeros(1,3,requires_grad=True)
    world=torch.tensor([[[0.,0.],[40.,0.],[0.,40.]]]);target=torch.zeros(1,3);known=torch.ones(1,3,dtype=torch.bool)
    geometry=torch.zeros(1,3,1,10);geometry[0,0,0,1]=.1
    roles=dict(role_valid=torch.tensor([True]),role_reference_index=torch.tensor([0]),role_direction=torch.tensor([[0.,1.]]))
    loss,eligible=compass_counterfactual_loss(real,opposite,world,target,known,geometry,roles)
    assert eligible.all();loss.backward();assert real.grad[0,0]<0 and opposite.grad[0,0]>0
    roles['role_direction']=-roles['role_direction']
    loss,eligible=compass_counterfactual_loss(real,opposite,world,target,known,geometry,roles)
    assert not eligible.any() and loss==0
