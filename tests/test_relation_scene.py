import numpy as np


def square(x,y,r=1):return [[x-r,y-r],[x+r,y-r],[x+r,y+r],[x-r,y+r]]


def test_explicit_reference_kept_and_context_gt_free():
    from graphnav.relation_bearing_scene import RelationSceneBuilder
    cat={'m':{'a':{'name':'Anchor','object_type':'Building','contour':square(0,0)},'far':{'name':'Other Name','object_type':'Building','contour':square(500,0)},**{str(i):{'name':'','object_type':'Car','contour':square(2+i*.1,0,.1)} for i in range(40)},'road':{'name':'Road','object_type':'TrafficRoad','contour':square(80,0,5)}}}
    b=RelationSceneBuilder(cat,max_nodes=12,radius_m=150)
    x=b.scene('m',[{'id':'a'},{'id':'far'}])
    assert 'a' in x['node_ids'] and 'far' in x['node_ids']
    assert 'road' in x['node_ids'] # cars cannot exhaust the context
    assert x['edges'].shape==(12,12,10)
    assert x['shape'].shape[0]==12
    assert x['edges'][x['node_ids'].index('road'),x['node_ids'].index('a'),0]>0
    assert np.isfinite(x['shape']).all()
    p=b.direction_kernel(x,['a','far'])
    np.testing.assert_allclose(p.sum(-1),1,atol=1e-5)
    assert p.shape==(2,12,8)


def test_boundary_topology_is_polygon_not_road_centerline():
    from graphnav.relation_bearing_scene import RelationSceneBuilder
    cat={'m':{'a':{'name':'Anchor','object_type':'TrafficRoad','contour':square(0,0,10)},'b':{'name':'B','object_type':'Building','contour':square(0,0,1)}}}
    b=RelationSceneBuilder(cat,max_nodes=4)
    x=b.scene('m',[{'id':'a'}]);i=x['node_ids'].index('b');j=x['node_ids'].index('a')
    assert x['edges'][i,j,3]==0 # boundary-region distance inside is zero
    assert x['edges'][i,j,7]>0 # candidate is contained by anchor
