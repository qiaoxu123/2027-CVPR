import copy
import numpy as np
import pytest
import torch
from experiments.language_object_graph.objects import ObjectGraph
from experiments.latent_object_binding.factors import between_points,Geometry
from experiments.latent_object_binding.language import parse
from experiments.latent_object_binding.inference import prepare,posterior
from experiments.latent_object_binding.resources import regions
from experiments.latent_object_binding.model import Matcher


def graph(ids=None):
    keys=ids or ['a','z','b','remote']
    objects={i:dict(name=name,object_type='Building',position=[x,0,0],contour=[[x-2,-2],[x+2,-2],[x+2,2],[x-2,2]]) for i,(x,name) in zip(keys,[(0,'White Church'),(20,''),(40,'Black Building'),(100,'')])}
    return ObjectGraph(objects,[-10,-20,110,20])


def public():
    grid=np.array([[x,y] for y in np.linspace(0,1,28) for x in np.linspace(0,1,28)])
    return dict(static_logits=np.zeros(784),grid_xy=grid,bounds=[-10,-20,110,20],pose_world=[20,0,50,0])


def test_real_polygon_between_symmetric():
    g=graph();xy=np.array([[20,0],[0,0],[100,0],[20,50]])
    a,b=g.shapes[g.byid['a']],g.shapes[g.byid['b']]
    assert np.array_equal(between_points(xy,a,b),between_points(xy,b,a))
    v=between_points(xy,a,b);assert v[0]>v[3] and v[1]==0 and v[2]<v[0]


def test_latent_binding_target_never_prefiltered():
    g=graph();p=prepare(g,'Find the small building between the White Church and the Black Building.',public())
    assert p['target_features'].shape[0]==len(g.ids)
    r=posterior(p);assert r['object_belief'].shape==(len(g.ids),) and r['heatmap'].shape==(784,) and torch.isfinite(r['heatmap']).all()
    assert torch.isclose(r['heatmap'].sum(),torch.tensor(1.))


def test_between_variable_permutation_preserves_goal():
    g=graph();text='Find the small building between the White Church and the Black Building.';l=parse(text,g);assert l['factors'][0]['kind']=='between'
    a=prepare(g,text,public(),language=l);b=copy.deepcopy(l);b['variables'][1],b['variables'][2]=b['variables'][2],b['variables'][1]
    for f in b['factors']:f['anchors']=[3-i if i in (1,2) else i for i in f['anchors']]
    rb=posterior(prepare(g,text,public(),language=b));ra=posterior(a)
    assert torch.allclose(ra['heatmap'],rb['heatmap'],atol=1e-7)


def test_ids_are_not_match_features():
    a=graph();b=graph(['43','91','7','11']);pa=prepare(a,'a small building',public());pb=prepare(b,'a small building',public())
    assert np.array_equal(pa['target_features'],pb['target_features'])
    assert torch.allclose(posterior(pa)['heatmap'],posterior(pb)['heatmap'])


def test_unknown_is_geometry_fallback():
    g=graph();l=dict(variables=[],entities=[],factors=[],hypotheses=[],confidence=0.)
    p=prepare(g,'unresolved',public(),language=l);r=posterior(p);assert r['gate']==0 and torch.allclose(r['heatmap'],torch.full((784,),1/784))


def test_unseen_and_occluded_region_remain_unknown():
    state=dict(metadata=dict(cutoff_step=3,episode_id='e',map_name='m'),regions=[dict(source_step=1,episode_id='e',map_name='m',object_id='1',native_min_side=50,depth_support=.5)])
    assert regions(state)=={}
    state['regions'][0]['source_step']=4
    with pytest.raises(ValueError):regions(state)


def test_gt_fields_rejected():
    p=public();p['target_object_id']='z'
    with pytest.raises(ValueError):prepare(graph(),'building',p)


def test_training_gradient_and_frozen_nine_parameter_budget():
    g=graph();p=prepare(g,'small building',public());model=Matcher();r=posterior(p,model.weights());loss=-torch.log(r['object_belief'][1]);loss.backward()
    assert model.theta.grad is not None and torch.isfinite(model.theta.grad).all()
    assert sum(x.numel() for x in model.parameters())==9


def test_reference_posterior_sums_target_likelihood():
    p=prepare(graph(),'a small building',public());n=len(graph().ids)
    p['target_features']=np.zeros((n,8),np.float32)
    p['reference_features']=np.zeros((2,8),np.float32)
    p['log_weights']=np.log(np.array([.5,.5],np.float32))
    p['assignments']=[{1:0},{1:1}]
    p['log_phi']=np.zeros((2,n+784),np.float32)
    p['log_phi'][0,:n]=-5;p['log_phi'][0,0]=0
    r=posterior(p)
    a=1+(n-1)*np.exp(-5);expected=np.array([a,n])/(a+n)
    assert np.allclose(r['assignment_posterior'].numpy(),expected,atol=1e-7)
    assert np.allclose(r['reference_prior_assignment_weights'].numpy(),[.5,.5])
    assert torch.isclose(sum(r['reference_binding_marginals']['1'].values()),torch.tensor(1.))


def test_coordinate_frame_translation_consistent():
    g=graph();text='building between White Church and Black Building';a=prepare(g,text,public());delta=np.array([123.,-51.]);objects=copy.deepcopy({o:i for o,i in zip(g.ids,g.objects)})
    for o in objects.values():
        o['position'][:2]=(np.array(o['position'][:2])+delta).tolist();o['contour']=(np.array(o['contour'])+delta).tolist()
    p=public();p['bounds']=(np.array(p['bounds'])+np.tile(delta,2)).tolist();p['pose_world'][:2]=(np.array(p['pose_world'][:2])+delta).tolist();g2=ObjectGraph(objects,p['bounds']);b=prepare(g2,text,p)
    assert torch.allclose(posterior(a)['heatmap'],posterior(b)['heatmap'],atol=1e-7)


def test_disconnected_reference_does_not_dilute_target_evidence():
    g=graph();text='a small building';l=parse(text,g);a=prepare(g,text,public(),language=l)
    e=copy.deepcopy(l['variables'][0]);e['role']='reference';e['named_ids']=[];e['object_type']='Car';l['variables'].append(e)
    b=prepare(g,text,public(),language=l)
    assert a['known_mass']==b['known_mass']
    assert torch.allclose(posterior(a)['heatmap'],posterior(b)['heatmap'])


def test_anonymous_between_permutation_with_beam_is_symmetric():
    g=graph();l=parse('building between two buildings',g)
    e=copy.deepcopy(l['variables'][0]);e['named_ids']=[];e['role']='reference';e['size']=None
    l['variables']=[l['variables'][0],copy.deepcopy(e),copy.deepcopy(e)];l['entities']=l['variables'];l['factors']=[dict(kind='between',subject=0,anchors=[1,2],subject_hypotheses=[dict(subject=0,probability=1.)])];l['hypotheses']=[dict(target=0,probability=1.)]
    ev={1:np.zeros((4,2),np.float32),2:np.zeros((4,2),np.float32)};ev[1][0,0]=.5;ev[2][2,0]=.7
    a=prepare(g,'test',public(),language=l,visual=ev,beam=4);ll=copy.deepcopy(l);ll['variables'][1],ll['variables'][2]=ll['variables'][2],ll['variables'][1];b=prepare(g,'test',public(),language=ll,visual={1:ev[2],2:ev[1]},beam=4)
    assert torch.allclose(posterior(a)['heatmap'],posterior(b)['heatmap'],atol=1e-7)


def test_same_public_named_entity_can_corefer():
    g=graph();l=parse('a building near White Church',g);same=copy.deepcopy(l['variables'][1]);l['variables'].append(same);l['factors'].append(dict(kind='near',subject=0,anchors=[2],subject_hypotheses=[dict(subject=0,probability=1.)]))
    p=prepare(g,'test',public(),language=l)
    assert p['support_mass']>0 and p['assignments'][0][1]==p['assignments'][0][2]


def test_half_cached_inputs_have_finite_matching_output():
    p=prepare(graph(),'small building',public())
    for key in ('target_features','reference_features','log_phi','log_weights'):p[key]=torch.tensor(p[key],dtype=torch.float16)
    assert torch.isfinite(posterior(p)['heatmap']).all()


def test_black_unobserved_evidence_and_shuffle_coverage():
    from experiments.latent_object_binding.visual import evidence,template
    g=graph();l=parse('white building',g);state=dict(metadata=dict(family_id='f',episode_id='e',map_name='m',cutoff_step=2),regions=[dict(source_step=1,episode_id='e',map_name='m',object_id='a',object_type='Building',native_min_side=32,depth_support=1.,view_key='v1'),dict(source_step=2,episode_id='e',map_name='m',object_id='z',object_type='Building',native_min_side=32,depth_support=1.,view_key='v2')])
    f=dict(bank={'v1':{'tight':torch.tensor([1.,0.])},'v2':{'tight':torch.tensor([0.,1.])}},pixel_zero={'image':torch.tensor([0.,1.])});templates={template(e):torch.tensor([1.,0.]) for e in l['variables']};pixels={'v1':{'color_scores':{'white':.9}}}
    black,bc=evidence(g,l,state,f,templates,pixels,'black');assert np.array_equal(black[0],np.zeros((4,2)))
    real,rc=evidence(g,l,state,f,templates,pixels,'real');shuffle,sc=evidence(g,l,state,f,templates,pixels,'shuffled');assert rc==bc==sc
    assert np.array_equal(real[0][[g.byid['b'],g.byid['remote']]],np.zeros((2,2)))


def test_reference_boundary_distance_handles_long_road():
    g=graph();g.shapes[g.byid['a']]=__import__('shapely').geometry.Polygon([[-1,-1],[39,-1],[39,1],[-1,1]])
    geom=Geometry(g,np.array([[20,0],[100,0]]))
    assert geom.object_relation('near',g.byid['z'],[g.byid['a']])==1.


def test_navigation_interface_keeps_784_candidate_grid():
    from graphnav.latent_object_binding import LatentObjectGrounder
    model=LatentObjectGrounder(graph());r=model('building between White Church and Black Building',public())
    assert r['goal_logits'].shape==(784,) and r['goal_heatmap'].shape==(28,28)
    assert torch.allclose(torch.softmax(r['goal_logits'],0).reshape(28,28),r['goal_heatmap'])
    bad=public();bad['target_world']=[20,0]
    with pytest.raises(ValueError):model('building',bad)
