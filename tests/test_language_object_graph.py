import numpy as np
import pytest
import torch
from experiments.language_object_graph.objects import ObjectGraph,public_catalog,FEATURE_DIM
from experiments.language_object_graph.language import parse,vector
from experiments.language_object_graph.reasoning import between,directional,mix
from experiments.language_object_graph.pipeline import forward,model_inputs,reconstruct,controller_field
from experiments.language_object_graph.model import LanguageGAT,goal_loss
from experiments.language_object_graph.rgb import CausalAttributes,color_measurements,frozen_phrase_evidence
from graphnav.full_localization import GoalField

def graph():
    raw={}
    for i,(x,name) in enumerate([(-30,'White Church'),(30,'Black Hall'),(0,'')]):
        raw[str(i)]={'id':i,'name':name,'object_type':'Building','position':[x,0,5],'dimension':[10,10,5],
                     'contour':[[x-5,-5],[x+5,-5],[x+5,5],[x-5,5]],'descriptions':['PRIVATE']}
    return ObjectGraph(public_catalog({'m':raw})['m'],[-50,-50,50,50])

def public():
    x,y=np.meshgrid(np.linspace(0,1,28),np.linspace(0,1,28));xy=np.stack([x.ravel(),y.ravel()],1)
    return dict(static_logits=np.zeros(784),grid_xy=xy,bounds=[-50,-50,50,50],pose_world=[0,0,50,0])

def test_public_whitelist_and_all_anonymous():
    g=graph();assert len(g.ids)==3;assert not any('descriptions' in x for x in g.objects);assert g.features.shape==(3,FEATURE_DIM)

def test_sparse_edges_exact_and_direction_reversal():
    g=graph();assert np.all(g.edge_attr[:,3]<=1)
    edges={tuple(e):a for e,a in zip(g.edge_index.T,g.edge_attr)}
    for (i,j),a in edges.items():assert np.allclose(a[:2],-edges[j,i][:2])

def test_ids_do_not_change_features_or_order():
    g=graph();o={str(99+i):dict(v,id=99+i) for i,v in enumerate(g.objects)};h=ObjectGraph(o,g.bounds)
    assert np.array_equal(g.features,h.features);assert np.array_equal(g.edge_index,h.edge_index)

def test_between_joint_not_near():
    p=np.array([[0,0],[-30,0],[30,0],[0,100]])
    v=between(p,[[-30,0],[30,0]]);assert v[0]>v[1] and v[0]>v[2] and v[0]>v[3]

def test_parse_between_and_named_anonymous():
    g=graph();l=parse('The small house between White Church and Black Hall.',g)
    assert l['entities'][0]['object_type']=='Building';assert len(l['factors'][0]['anchors'])==2
    assert all(l['entities'][i]['named_ids'] for i in l['factors'][0]['anchors'])

def test_attribute_scope():
    l=parse('A gray building with a white car beside it.',graph());assert l['entities'][0]['colors']==('gray',)
    assert not l['entities'][-1]['object_type']

def test_relative_ambiguity_retained():
    l=parse('A house left of Black Hall',graph());assert len(l['factors'][0]['frame_hypotheses'])==3
    assert directional('east',np.array([[20,0],[-20,0]]),np.zeros(2),0)[0]>.5

def test_raw_public_inference_rejects_target_keys():
    p=public();p['target_id']=1
    with pytest.raises(ValueError):forward(graph(),'house',p)

def test_zero_mixture_exact_ties():
    p=public();assert np.array_equal(mix(p['static_logits'],np.ones(784)/784,0),p['static_logits'])

def test_continuous_all_candidates_and_unknown_visual():
    g=graph();p=public();r=forward(g,'house between White Church and Black Hall',p)
    assert r['heatmap'].shape==(784,) and np.all(r['heatmap']>0);assert np.isclose(r['heatmap'].sum(),1)
    h=forward(g,'house between White Church and Black Hall',p,visual_evidence=np.zeros(3));assert np.allclose(r['logits'],h['logits'])

def test_gat_gradient_public_contract_and_full_reconstruction():
    torch.set_num_threads(1);g=graph();r=forward(g,'house near White Church',public());p,idx=model_inputs(g,r);m=LanguageGAT();o=m(p);full=reconstruct(r,idx,o['delta']);positive=torch.zeros(784,dtype=torch.bool);positive[400]=True
    goal_loss(full,positive).backward();assert torch.isfinite(m.output[-1].weight.grad).all();assert m.output[-1].weight.grad.abs().sum()>0
    with pytest.raises(ValueError):m({**p,'target_id':torch.tensor(1)})

def test_outside_subgraph_preserved():
    r=forward(graph(),'house',public());x=reconstruct(r,np.array([0]),torch.tensor([0.]));assert np.allclose(x['object_belief'].numpy(),r['object_belief'],atol=1e-6)

def test_controller_topk_context_alignment_and_immutability():
    logits=torch.zeros(1,4);ids=torch.tensor([[0,1]]);xy=torch.arange(8).reshape(4,2).float();dense=torch.arange(12).reshape(1,4,3).float()
    field=GoalField(logits,ids,xy[None,:2],dense[:,:2],dense_context=dense)
    new=controller_field(field,torch.tensor([[0.,1.,2.,3.]]),xy)
    assert new.candidate_ids.tolist()==[[3,2]];assert torch.equal(new.candidate_context,dense[:,[3,2]]);assert field.candidate_ids.tolist()==[[0,1]]

def test_color_pixels_not_truth():
    bgr=np.full((32,32,3),255,np.uint8);c=color_measurements(bgr,np.ones((32,32),bool));assert c['fractions']['white']==1;assert 'unverified' in c['truth']

def test_future_rejected_and_unknown_no_evidence():
    a=CausalAttributes()
    with pytest.raises(ValueError):a.observe({},graph(),2,1)
    assert np.array_equal(a.evidence({'colors':['white']},3),np.zeros(3))

def test_black_shuffle_same_eligibility_causality():
    g=graph();p=np.full((224,224,3),220,np.uint8);obs=dict(bgr=p,depth_valid=np.ones((224,224),bool),footprint_world=np.array([[-50,50],[50,50],[50,-50],[-50,-50]]))
    controls=[CausalAttributes() for _ in range(3)]
    for a,c in zip(controls,['real','black','shuffled']):a.observe(obs,g,0,0,control=c)
    assert controls[0].nodes.keys()==controls[1].nodes.keys()==controls[2].nodes.keys();assert len(controls[0].nodes)>0
    assert controls[1].nodes[0]['foreground']['fractions']['black']==1
    with pytest.raises(ValueError):controls[0].observe(obs,g,0,0)

def test_cached_embeddings_unknown_mask_same_dimension():
    s,use=frozen_phrase_evidence(np.eye(2),np.array([1,0]),[True,False],[1,1]);assert s.tolist()==[1,0];assert use.tolist()==[True,False]
    with pytest.raises(ValueError):frozen_phrase_evidence(np.eye(2),np.ones(3),[True,True],[1,1])

def test_reference_rgb_updates_distribution_and_unknown_zero():
    from experiments.language_object_graph.reasoning import candidates
    g=graph();r=forward(g,'house near a white building',public());anchor=r['language']['entities'][1]
    prior=r['kernels']@r['static'];ids,w=candidates(anchor,g,prior);ev=np.zeros(3);ev[ids[-1]]=1
    new_ids,new_w=candidates(anchor,g,prior,visual=ev)
    assert np.array_equal(ids,new_ids);assert new_w[-1]>w[-1]
    _,zero=candidates(anchor,g,prior,visual=np.zeros(3));assert np.allclose(zero,w)

def test_observed_forward_checks_causal_memory_and_controls():
    from experiments.language_object_graph.pipeline import observed_forward
    memory=CausalAttributes();memory.nodes={0:dict(step=3,control='real',quality=1,foreground=None)}
    with pytest.raises(ValueError):observed_forward(graph(),'white house',public(),memory,weight=.5,cutoff_step=2)
    with pytest.raises(ValueError):observed_forward(graph(),'white house',public(),memory,weight=.5,cutoff_step=3,control='black')
    r=observed_forward(graph(),'white house',public(),memory,weight=.5,cutoff_step=3)
    base=forward(graph(),'white house',public(),method='A2',weight=.5);assert np.allclose(r['logits'],base['logits'])


def test_gat_messages_have_gradient_after_zero_initialization_step():
    g=graph();r=forward(g,'house near White Church',public());p,idx=model_inputs(g,r);m=LanguageGAT();optimizer=torch.optim.Adam(m.parameters(),lr=.001)
    for _ in range(2):
        optimizer.zero_grad();out=m(p);full=reconstruct(r,idx,out['delta']);positive=torch.zeros(784,dtype=torch.bool);positive[400]=True;goal_loss(full,positive).backward();optimizer.step()
    assert m.attention.grad is not None and torch.isfinite(m.attention.grad).all() and m.attention.grad.abs().sum()>0

def test_frozen_attributes_abstain_without_calibration_and_cache_content():
    from experiments.language_object_graph.evaluate_rgb import FrozenAttributes
    class Encoder:
        calls=0
        def images(self,images):
            self.calls+=len(images)
            return np.array([[float(x.mean()),1] for x in images],np.float32)
        def texts(self,text):return np.array([[1.,0.] for _ in text],np.float32)
    g=graph();encoder=Encoder();a=FrozenAttributes(encoder,'synthetic-test-only',threshold=None)
    obs=dict(bgr=np.full((224,224,3),220,np.uint8),depth_valid=np.ones((224,224),bool),footprint_world=np.array([[-50,50],[50,50],[50,-50],[-50,-50]]))
    a.observe(obs,g,0,1);assert len(a.nodes)>0;assert not a.evidence({'colors':['white'],'shape':None,'object_type':'Building','unknown':[]},3).any()
    n=len(a.images);a.observe(obs,g,1,1,control='black');assert len(a.images)>n
    with pytest.raises(ValueError):a.observe(obs,g,2,1)
