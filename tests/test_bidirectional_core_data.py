"""Tests for reverse-time supervision and isolation, not network implementation."""
import numpy as np
from shapely.geometry import Polygon
from experiments.bidirectional_core.data import reverse_trajectory,fixed_prefixes,ground_start,geometry_description,internal_partitions

def test_reversal_preserves_real_xyz_and_recomputes_stationary_heading():
    src=[[0,0,20,0,0],[0,0,21,0,0],[10,0,22,0,0],[10,10,23,0,0]]
    rev=reverse_trajectory(src)
    np.testing.assert_array_equal(rev[:,:3],np.asarray(src)[::-1,:3])
    np.testing.assert_allclose(rev[:,3],[-np.pi/2,np.pi,np.pi,np.pi])

def test_fixed_prefixes_have_actual_endpoint_and_no_future_index():
    assert fixed_prefixes(9)==[0,2,4,6,8]
    assert fixed_prefixes(2)==[0,1]

def entry(oid,poly,name=''):
    p=Polygon(poly)
    return dict(id=str(oid),object_type='Building',name=name,polygon=p,center=np.asarray(p.centroid.coords[0]),item=dict(name=name,object_type='Building',contour=poly))

def test_overlap_and_boundary_are_unknown_not_arbitrary_identity():
    a=entry(1,[[0,0],[20,0],[20,20],[0,20]])
    b=entry(2,[[5,5],[25,5],[25,25],[5,25]])
    assert ground_start([10,10],[a,b])[0] is None
    assert ground_start([0,10],[a])[0] is None
    assert ground_start([1,10],[a])[0] is None
    assert ground_start([10,10],[a])[0] is a

def test_generated_return_language_uses_geometry_and_preserves_multi_positive():
    anchor=entry(1,[[-5,-5],[5,-5],[5,5],[-5,5]],'Church')
    a=entry(2,[[45,-5],[55,-5],[55,5],[45,5]])
    b=entry(3,[[45,0],[55,0],[55,10],[45,10]])
    d=geometry_description(a,[anchor,a,b])
    assert set(d['label_positive_object_ids'])=={'2','3'}
    assert 'east' in d['instruction'] and '50 meters' in d['instruction']
    assert not set(d['label_positive_object_ids'])&set(d['label_negative_object_ids'])
    point=geometry_description(None,[anchor],start_xy=[50,0])
    assert not point['visual_pair_eligible'] and 'launch point' in point['instruction']

def test_family_shared_object_and_visited_region_cannot_cross_internal_split():
    def r(f,target,path):return dict(family_id=f,map_name='m',target_id=target,start_object_id=None,trajectory=np.asarray(path,float))
    records=[r('a','1',[[0,0],[10,0]]),r('b','1',[[500,0],[510,0]]),r('c','2',[[500,0],[510,0]])]
    parts,stats=internal_partitions(records)
    assert len(set(parts.values()))==1
    assert stats['node_overlap']==0 and stats['components']==1
