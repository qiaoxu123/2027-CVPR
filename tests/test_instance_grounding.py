import numpy as np
import pytest
import torch
from experiments.instance_grounding.data import update_memory,choose
from experiments.instance_grounding.model import InstanceVEN,loss
from experiments.instance_grounding.dataset import collate,training_unit

def inputs(n=3):
    return dict(full=torch.randn(1,1024),phrase=torch.randn(1,1024),images=torch.randn(1,n,1024),context=torch.randn(1,n,1024),patch_tokens=torch.randn(1,n,16,1024),language_tokens=torch.randn(1,4,1024),language_valid=torch.ones(1,4,dtype=torch.bool),quality=torch.ones(1,n,3),valid=torch.ones(1,n,dtype=torch.bool),relative=torch.zeros(1,n,n+1,3))

def test_private_labels_rejected():
    model=InstanceVEN();p=inputs();p['target_object_id']=torch.tensor([1])
    with pytest.raises(ValueError):model(p)

def test_empty_observations_unknown_and_finite():
    model=InstanceVEN();p=inputs();p['valid'].zero_();s,prob,tri=model(p)
    assert torch.isfinite(s).all() and torch.isfinite(tri).all()
    assert torch.equal(prob,torch.tensor([[[0.,0.,1.]]]).expand_as(prob))
    assert (s==0).all()

def test_nonfinite_rejected():
    model=InstanceVEN();p=inputs();p['images'][0,0,0]=torch.nan
    with pytest.raises(ValueError):model(p)

def test_candidate_permutation_equivariance():
    torch.manual_seed(0);model=InstanceVEN().eval();p=inputs();order=torch.tensor([2,0,1]);q={k:(v[:,order] if k in ['images','context','patch_tokens','quality','valid'] else v) for k,v in p.items()}
    a,ap,_=model(p);b,bp,_=model(q)
    assert torch.allclose(a[:,order],b,atol=1e-5) and torch.allclose(ap[:,order],bp,atol=1e-5)

def region(oid,step,side):return dict(object_id=oid,source_step=step,depth_support=1,native_min_side=side,quality=[1,1,1])

def test_future_rejected_and_old_view_retained():
    m={};update_memory(m,[region('a',2,12)],2);update_memory(m,[region('a',3,8)],3)
    assert m['a']['source_step']==2
    with pytest.raises(ValueError):update_memory(m,[region('a',5,40)],4)

def test_current_objects_not_dropped_by_history_cap():
    m={str(i):region(str(i),1,i+8) for i in range(5)}
    assert choose(m,['0'],2)[0]['object_id']=='0'

def test_unknown_not_forced_negative_in_ranking():
    p=inputs();model=InstanceVEN();s,prob,tri=model(p);pos=torch.tensor([[True,False,False]]);judged=torch.tensor([[True,True,False]])
    val=loss(s,tri,pos,judged,p['valid'],p['quality']);assert torch.isfinite(val);val.backward();assert any(x.grad is not None for x in model.parameters())

def test_absent_target_uses_only_unknown_judgment():
    p=inputs();model=InstanceVEN();s,_,tri=model(p);mask=torch.zeros(1,3,dtype=torch.bool);val=loss(s,tri,mask,mask,p['valid'],p['quality'])
    expected=.2*torch.nn.functional.cross_entropy(tri[p['valid']],torch.full((3,),2,dtype=torch.long))
    assert torch.allclose(val,expected)

def test_prefixes_one_training_family():
    rows=[dict(metadata=dict(family_id='x',cutoff_step=i)) for i in range(5)]
    assert len(training_unit(rows))==1 and training_unit(rows)[0]['metadata']['cutoff_step']==4

def test_relation_word_not_negative_without_named_anchor():
    from experiments.instance_grounding.relations import relation_mismatches
    a=dict(object_id='a',name='',center_world=[20,0]);b=dict(object_id='b',name='',center_world=[-20,0])
    assert relation_mismatches('building near another building',[a,b],'a')==set()

def test_same_color_identity_requires_actual_relation_contradiction():
    from experiments.instance_grounding.relations import relation_mismatches
    a=dict(object_id='a',name='',center_world=[20,0]);b=dict(object_id='b',name='',center_world=[-20,0]);anchor=dict(object_id='c',name='Depot',center_world=[0,0])
    assert relation_mismatches('building east of Depot',[a,b,anchor],'a')=={'b'}

def fixture_row():
    r=dict(metadata=dict(roi_objects=[dict(object_type='Building',bbox=[0,0,10,10]),dict(object_type='Building',bbox=[20,0,30,10]),dict(object_type='Car',bbox=[0,20,10,30])]),public=dict(roi_valid=torch.ones(3,dtype=torch.bool),roi_quality=torch.ones(3,3)),labels=dict(positive_roi_mask=torch.tensor([True,False,False]),semantic_valid_mask=torch.tensor([True,False,False])),features=dict(tight=torch.stack([torch.full((1024,),float(i)) for i in range(3)]),context=torch.stack([torch.full((1024,),float(i+10)) for i in range(3)])),patch_tokens=torch.stack([torch.full((16,1024),float(i)) for i in range(3)]),texts=dict(full=torch.ones(1024),phrase=torch.ones(1024)),language_tokens=torch.ones(4,1024),language_valid=torch.ones(4,dtype=torch.bool),pixel_zero=dict(image=torch.full((1024,),7.),patch_tokens=torch.full((16,1024),8.)))
    return r

def test_zero_rgb_uses_encoded_black_not_zero_vector():
    r=fixture_row();p,_,_=collate([r],'zero');assert (p['images']==7).all() and (p['patch_tokens']==8).all();assert r['features']['tight'][0,0]==0

def test_shuffle_same_class_and_preserves_other_inputs():
    r=fixture_row();a,_,_=collate([r]);b,_,_=collate([r],'shuffled')
    assert b['images'][0,:,0].tolist()==[1,0,2]
    for k in ['full','phrase','quality','valid','language_tokens','language_valid','relative']:assert torch.equal(a[k],b[k])

def fusion_fixture():
    from experiments.instance_grounding.evaluate import public_fusion
    state=dict(public=dict(static_logits=[0.]*784,bounds=[0.,0.,100.,100.],grid_xy=[[(x+.5)/28,(y+.5)/28] for y in range(28) for x in range(28)]))
    regions=[dict(actual_camera_pose=[50.,50.,20.,0.],ground_level=0,contour=[[46,46],[54,46],[54,54],[46,54]],depth_support=1)]
    cfg=dict(temperature=1,threshold=.8,gain=.025)
    return public_fusion,state,regions,cfg

def test_unknown_confidence_cannot_change_any_grid():
    fuse,s,r,c=fusion_fixture();pred=dict(score=torch.tensor([1.]),tri=torch.tensor([[-10.,-10.,10.]]));out,w,res,accepted=fuse(s,r,pred,c);assert (res==0).all() and accepted is None

def test_accepted_residual_bounded_and_unobserved_exact():
    fuse,s,r,c=fusion_fixture();pred=dict(score=torch.tensor([1.]),tri=torch.tensor([[10.,-10.,-10.]]));out,w,res,accepted=fuse(s,r,pred,c);assert accepted==0 and abs(res).max()<=.025
    assert np.array_equal(out[w.sum(0)==0],np.array(s['public']['static_logits'])[w.sum(0)==0])

def test_aerial_aliased_primary_proposals_iou():
    from experiments.instance_grounding.aerial_identity import overlap
    assert overlap([0,0,10,10],[0,0,10,10])==1 and overlap([0,0,10,10],[10,10,20,20])==0

@pytest.mark.parametrize('location',[[50,50],[2,2],[99,99],[45,10]])
def test_fast_centroid_equals_frozen_association(location):
    from experiments.instance_grounding.association import centroid_support
    from experiments.visual_goal_association.association import associate
    grid=np.array([[(x+.5)/28,(y+.5)/28] for y in range(28) for x in range(28)]);bounds=[0,0,100,100];x,y=location;poly=[[x-1,y-1],[x+1,y-1],[x+1,y+1],[x-1,y+1]];fov=[[0,0],[60,0],[60,60],[0,60]]
    assert np.array_equal(centroid_support(poly,fov,bounds,grid),associate(poly,fov,bounds,grid,'centroid'))

def test_relative_geometry_is_fixed_in_rgb_controls():
    r=fixture_row();a,_,_=collate([r]);b,_,_=collate([r],'shuffled');z,_,_=collate([r],'zero')
    assert torch.equal(a['relative'],b['relative']) and torch.equal(a['relative'],z['relative'])

def test_relative_attention_empty_memory_and_gradients():
    model=InstanceVEN(relative_attention=True);p=inputs();s,prob,tri=model(p);assert torch.isfinite(s).all();s.sum().backward();assert model.relation_coefficients.weight.grad is not None
    p['valid'].zero_();s,prob,_=model(p);assert torch.isfinite(s).all() and (prob[...,2]==1).all()

def test_relative_attention_permutation_equivariance():
    torch.manual_seed(12);model=InstanceVEN(relative_attention=True).eval();torch.nn.init.normal_(model.relation_coefficients.weight,std=.02);p=inputs();p['relative']=torch.randn(1,3,4,3);order=torch.tensor([2,0,1]);q={k:(v[:,order] if k in ['images','context','patch_tokens','quality','valid'] else v) for k,v in p.items()};q['relative']=p['relative'][:,order][:,:,[2,0,1,3]]
    a,ap,_=model(p);b,bp,_=model(q);assert torch.allclose(a[:,order],b,atol=1e-5) and torch.allclose(ap[:,order],bp,atol=1e-5)
