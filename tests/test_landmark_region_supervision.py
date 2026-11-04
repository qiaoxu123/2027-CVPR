import numpy as np
from graphnav.landmark_region_supervision import named_union,weak_region_labels,public_type


def test_region_proxies_preserve_unknown_for_unsupported_or_boundary_windows():
    region=named_union([[[0,0],[10,0],[10,10],[0,10]]])
    windows=np.array([[[2,2],[4,2],[4,4],[2,4]],[[12,12],[14,12],[14,14],[12,14]],
        [[9,2],[11,2],[11,4],[9,4]],[[2,2],[4,2],[4,4],[2,4]]])
    labels,_=weak_region_labels(windows,region,np.array([True,True,True,False]))
    assert labels.tolist()==[1,0,-1,-1]


def test_multifragment_roads_and_mixed_type_abstention():
    assert public_type(['TrafficRoad','Footpath'])=='Road'
    assert public_type(['Building','TrafficRoad']) is None
    assert public_type(['Building'])=='Building'
