import numpy as np
import pytest
from experiments.rgb_feasibility.core import percentile,auc,reliability,risk_curve,offline_residual,public_relations,compose_scores
from experiments.rgb_feasibility.run import query_spec

def test_tie_percentiles_do_not_create_identity_preference():
 assert np.array_equal(percentile([1,1,1]),[.5,.5,.5])
 assert np.array_equal(percentile([2,1,2]),[.75,0,.75])
 assert auc([1,0,1,0],[1,1,1,1])==.5
 assert auc([1,0],[2,1])==1
 assert auc([1,1],[2,1]) is None

def test_ece_and_brier_account_for_wrong_high_confidence():
 a=reliability([1,0],[.8,.8]);assert a['ece']==pytest.approx(.3);assert a['brier']==pytest.approx(.34)
 with pytest.raises(ValueError):reliability([1],[1.2])

def test_prefixes_cannot_supply_family_safety():
 rows=[dict(family_id='one',eligible=True,correct=True,match=1) for _ in range(100)]
 v=risk_curve(rows,'match',[.8])[0];assert v['n']==100 and v['families']==1 and not v['supported_safe_interval']
 rows=[dict(family_id=str(i),eligible=True,correct=True,match=1) for i in range(40)]
 assert risk_curve(rows,'match',[.8])[0]['supported_safe_interval']
 rows[0]['correct']=False;assert risk_curve(rows,'match',[.8])[0]['family_any_error_risk']==pytest.approx(.025)

def test_theoretical_residual_needs_explicit_offline_flag():
 s=np.array([.3,.1,0]);w=np.array([[0,1,0]])
 with pytest.raises(ValueError):offline_residual(s,w,0,.5)
 u,res=offline_residual(s,w,0,.5,offline_only=True);assert u[1]>.3 and np.array_equal(u[[0,2]],s[[0,2]])
 assert np.array_equal(offline_residual(s,w,None,8,offline_only=True)[0],s)
 with pytest.raises(ValueError):offline_residual(s,w,1,8,offline_only=True)

def test_relation_rules_do_not_use_gt_identity():
 def region(x,name):return dict(name=name,center_world=[x,0],contour=[[x-1,-1],[x+1,-1],[x+1,1],[x-1,1]])
 anchor=region(0,'Central Church');regions=[region(30,''),region(-30,''),anchor]
 s,known,used=public_relations('the building east of Central Church',regions)
 assert used and s[0]==1 and s[1]==-1 and not known[2]
 s,known,used=public_relations('the building near the church',regions);assert not used and not known.any()

def test_attribute_swap_cannot_relabel_reference_car_white():
 spec=query_spec('A gray building with a white car beside it.')
 assert spec['attributes'].get('building_color')=='gray';assert spec['swap']['phrase']=='a white building'
 assert 'car' not in spec['swap']['phrase']

def test_combination_is_fixed_and_unknown_relations_leave_scores():
 rgb=np.array([1,2,3]);g=np.array([3,2,1]);z=np.zeros(3)
 assert np.array_equal(compose_scores(rgb,g,z,'attribute_spatial','rgb',False),percentile(rgb))
 assert np.array_equal(compose_scores(rgb,g,z,'full','combined',False),[.5,.5,.5])

def test_identity_ties_and_offline_labels_serialize_without_index_wins():
 import json,torch
 from experiments.rgb_feasibility.run import metric_record,offline_labels
 r=dict(state=dict(public=dict(bounds=[0,0,28,28],grid_xy=np.zeros((784,2)).tolist(),static_logits=np.zeros(784).tolist())),sealed=dict(target_world=[0,0],target_object_id='a'),metadata=dict(sample_id='s',family_id='f',map_name='m',cutoff_step=0,roi_objects=[dict(object_id='a',object_type='Building',native_min_side=32,source_step=0),dict(object_id='b',object_type='Building',native_min_side=32,source_step=0)]),labels=dict(identity_positive=torch.tensor([True,False]),target_credible=True,same_class=torch.tensor([True,True]),same_color=torch.tensor([True,True]),attribute_judged=torch.tensor([True,True]),attribute_positive=torch.tensor([True,False])))
 labels=offline_labels(r);assert isinstance(labels['wrong'],bool)
 result=metric_record(r,np.array([1.,1.]),'full',labels);json.dumps(result,allow_nan=False)
 assert result['metrics']['identity_r1']==.5 and result['metrics']['target_rank']==1.5
