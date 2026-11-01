import inspect
import numpy as np
import torch
from graphnav.native_multireference import (MultiReferenceRanker,multireference_descriptors,
                                           auxiliary_reference_loss,belief_training_loss)
from graphnav.semantic_geometry import SpatialRelation,parse_spatial_relations
from test_native_relation_grounding import graph,batch


def public_batch():
    g=graph('between')
    e=dict(g['entities'][0]);e['name']='Library';e['geometry_m']=e['geometry_m']+np.array([100,0])
    d=dict(e);d['name']='Real distractor';d['geometry_m']=d['geometry_m']+np.array([0,10])
    g['entities'] += [e,d]
    g['relations']=[SpatialRelation('between',((0,),(1,)))]
    x=batch();data=multireference_descriptors(np.array([[50,90],[50,10],[10,50]],np.float32),g)
    for key in ['relation_features','relation_types','relation_valid','native_null']:x.pop(key)
    x.update({k:v[None] for k,v in data.items()})
    return x


def test_true_between_anchors_and_actual_distractor_are_distinct():
    x=public_batch()
    assert x['reference_valid'].sum()==3
    assert x['reference_binding'].sum()==2
    assert not torch.equal(x['reference_features'][:,:,0],x['reference_features'][:,:,1])
    assert 'target_world' not in inspect.signature(multireference_descriptors).parameters
    assert 'target_world' not in inspect.signature(MultiReferenceRanker.forward).parameters


def test_real_multislot_attention_has_query_key_gradients_from_public_binding():
    x=public_batch();model=MultiReferenceRanker()
    out=model(**{k:v for k,v in x.items() if k!='reference_binding'})
    attn,geometry=auxiliary_reference_loss(out,x)
    (attn+.1*geometry).backward()
    assert model.query.weight.grad.abs().sum()>0
    assert model.key.weight.grad.abs().sum()>0
    assert model.geometry_head.weight.grad.abs().sum()>0
    assert not out['attention'][0,2].any()
    assert not out['attention'][:,:,3:].any()


def test_no_relation_and_disabled_preserve_native():
    x=public_batch();model=MultiReferenceRanker()
    with torch.no_grad():model.head[-1].bias.fill_(1.)
    kw={k:v for k,v in x.items() if k!='reference_binding'}
    assert torch.equal(model(**kw,enabled=False)['logits'],x['native_logits'])
    kw['reference_valid']=torch.zeros_like(x['reference_valid'])
    assert torch.equal(model(**kw)['logits'],x['native_logits'])


def test_compound_directions_and_world_translation():
    g=graph();g['relations']=parse_spatial_relations('north-east of The School',g['entities'])
    xy=np.array([[100,100],[0,0]],np.float32)
    before=multireference_descriptors(xy,g)['reference_features']
    for e in g['entities']:e['geometry_m']+=1000
    after=multireference_descriptors(xy+1000,g)['reference_features']
    assert torch.equal(before,after)
    assert before[0,0,4]>0 and before[0,0,5]>0


def test_belief_loss_is_finite_without_a_positive_grid_and_gt_only_in_loss():
    x=dict(grid_world=torch.tensor([[[0.,0.],[40.,0.]]]),target_world=torch.tensor([[1000.,0.]]))
    updated=torch.tensor([[1.,0.]],requires_grad=True)
    losses=belief_training_loss(updated,updated.detach(),x,torch.tensor([True]))
    assert all(torch.isfinite(v) for v in losses)
    sum(losses).backward()


def test_reference_slot_permutation_and_private_target_mutation_do_not_affect_forward():
    x=public_batch();model=MultiReferenceRanker()
    with torch.no_grad():model.head[-1].weight.normal_(0.,.05)
    public={k:v for k,v in x.items() if k!='reference_binding'}
    expected=model(**public)['logits']
    order=torch.tensor([2,0,1,3,4,5,6,7]);changed=dict(public)
    changed['reference_features']=public['reference_features'][:,:,order]
    changed['reference_types']=public['reference_types'][:,order]
    changed['reference_valid']=public['reference_valid'][:,order]
    assert torch.allclose(model(**changed)['logits'],expected,atol=1e-6)
    x['target_world']=torch.tensor([[1e9,1e9]])
    assert torch.equal(model(**public)['logits'],expected)


def test_v2_public_preparation_ignores_private_target_coordinates():
    from scripts.train_native_preserving_relation_gate import prepare
    class Static:
        records=[dict(map_name='public',instruction='north of The School',world_bounds=[0,0,100,100],
                      landmarks=[dict(name='The School',contour=[[.4,.4],[.6,.4],[.6,.6],[.4,.6]])])]
        def _prepared(self,index):return graph(),None
    row=dict(map_name='public',instruction='north of The School',query_world=torch.tensor([[50.,90.],[50.,10.]]),
             labels=dict(target_world=torch.tensor([50.,90.])))
    before=prepare([row],Static())[0]
    changed=dict(row,labels=dict(target_world=torch.tensor([-1e9,1e9])))
    after=prepare([changed],Static())[0]
    for key in ['reference_features','reference_types','reference_valid','reference_binding']:
        assert torch.equal(before[key],after[key])


def test_native_correct_preservation_penalizes_relation_peak_damage():
    x=dict(grid_world=torch.tensor([[[0.,0.],[40.,0.]]]),target_world=torch.tensor([[0.,0.]]))
    updated=torch.tensor([[0.,3.]],requires_grad=True)
    native=torch.tensor([[3.,0.]])
    loc,kl,safe=belief_training_loss(updated,native,x,torch.tensor([True]))
    assert loc>0 and kl>0 and safe>0
    safe.backward()
    assert updated.grad[0,0]<0 and updated.grad[0,1]>0
