import numpy as np
import pytest
import torch
from graphnav.landmark_navigation_diagnostic import public_region,oracle_goal_xy,polygon_trace_metrics,oracle_goal_field
from graphnav.full_localization import GoalField


def test_long_road_uses_polygon_distance_instead_of_centroid():
    region=public_region([[[0,0],[1000,0],[1000,5],[0,5]]])
    goal=oracle_goal_xy(region,[10,50])
    assert np.allclose(goal,[10,5])
    result=polygon_trace_metrics([[10,50,40],[10,10,40]],region,'policy_stop')
    assert result['SR']==1 and result['NE']==5
    assert np.linalg.norm(np.array([10,10])-np.array([500,2.5]))>400


def test_homonymous_alternatives_remain_union_and_bounds_clip():
    polys=[[[0,0],[10,0],[10,10],[0,10]],[[100,0],[110,0],[110,10],[100,10]]]
    region=public_region(polys,[-1,-1,105,12])
    assert np.allclose(oracle_goal_xy(region,[120,5]),[105,5])
    assert polygon_trace_metrics([[50,5],[102,5]],region,'horizon')['NE']==0
    with pytest.raises(ValueError):public_region(polys,[200,200,300,300])


def test_oracle_requires_explicit_mode_and_preserves_source_context():
    uv=torch.tensor([[[0.,0.],[1.,0.],[0.,1.],[1.,1.]]])
    dense=torch.randn(1,4,6)
    field=GoalField(torch.zeros(1,4),torch.tensor([[0,1]]),uv[:,:2].clone(),dense[:,:2].clone(),dense_context=dense)
    bounds=torch.tensor([[0.,0.,100.,100.]])
    with pytest.raises(PermissionError):oracle_goal_field(field,uv,bounds,[5,10])
    result=oracle_goal_field(field,uv,bounds,[5,10],diagnostic=True)
    assert torch.allclose(result.candidate_xy[0,0],torch.tensor([.05,.1]))
    assert torch.equal(result.candidate_context[0,0],dense[0,0])
    assert torch.equal(field.belief_logits,torch.zeros(1,4))
