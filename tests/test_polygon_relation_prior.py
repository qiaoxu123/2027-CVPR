import numpy as np
import pytest
from shapely.geometry import Polygon
from scripts.polygon_relation_prior import relation_prior


def test_topology_and_metric_support():
    poly=Polygon([(-10,-10),(10,-10),(10,10),(-10,10)])
    xy=np.array([[0,0],[10,0],[20,0],[70,0]],float)
    inside,mi=relation_prior(xy,poly,'inside')
    outside,mo=relation_prior(xy,poly,'outside')
    near,mn=relation_prior(xy,poly,'near')
    assert mi.tolist()==[True,True,False,False]
    assert mo.tolist()==[False,False,True,True]
    assert mn.tolist()==[True,True,True,False]
    assert inside[0]>inside[-1] and outside[-1]>outside[0] and near[0]>near[-1]


def test_direction_abstains_without_verified_georeference():
    p=Polygon([(0,0),(1,0),(1,1),(0,1)])
    with pytest.raises(ValueError,match='georeference'):
        relation_prior(np.array([[0.,0.]]),p,'north')


def test_grid_without_inside_support_remains_unknown():
    p=Polygon([(0,0),(1,0),(1,1),(0,1)])
    _,mask=relation_prior(np.array([[2.,2.],[3.,3.]]),p,'inside')
    assert not mask.any()  # do not manufacture a positive grid cell


def test_all_polygons_and_holes_used():
    from shapely.geometry import MultiPolygon
    a=Polygon([(0,0),(10,0),(10,10),(0,10)],holes=[[(3,3),(7,3),(7,7),(3,7)]])
    b=Polygon([(100,100),(110,100),(110,110),(100,110)])
    _,inside=relation_prior(np.array([[1,1],[5,5],[105,105]],float),MultiPolygon([a,b]),'inside')
    assert inside.tolist()==[True,False,True]


def test_region_margin_has_nonzero_finite_gradient():
    import torch
    from torch.nn import functional as F
    logits=torch.zeros(1,4,requires_grad=True)
    region=torch.tensor([[True,True,False,False]])
    mass=(logits.softmax(-1)*region).sum(-1)
    loss=F.relu(.2-mass+(1-mass)).mean();loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad[region]<0).all() and (logits.grad[~region]>0).all()
