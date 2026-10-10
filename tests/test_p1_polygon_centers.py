import numpy as np
from shapely.geometry import Polygon,Point
from graphnav.bidirectional import _centroid,generate_pairs


def test_p1_area_center_is_independent_of_vertex_density_and_inside_concavity():
    regular=[[0,0],[10,0],[10,10],[0,10]]
    dense=[[0,0],[2,0],[4,0],[8,0],[10,0],[10,10],[0,10]]
    assert np.allclose(_centroid(regular),_centroid(dense))
    concave=[[0,0],[10,0],[10,2],[2,2],[2,10],[0,10]]
    assert not Polygon(concave).covers(Polygon(concave).centroid)
    assert Polygon(concave).covers(Point(_centroid(concave)))


def test_p1_rejects_cardinal_centers_embedded_in_other_polygon():
    objects={'map':{'outer':dict(name='outer',object_type='Building',contour=[[10,10],[150,10],[150,150],[10,150]]),
        'inner':dict(name='inner',object_type='Building',contour=[[110,70],[120,70],[120,80],[110,80]])}}
    assert list(generate_pairs(objects,{'map':[0,0,200,200]},['map']))==[]


def test_p1_inverse_targets_stay_on_actual_source_and_record_boundary_clearance():
    objects={'map':{'a':dict(name='a',object_type='Building',contour=[[30,30],[50,30],[50,50],[30,50]]),
        'b':dict(name='b',object_type='Building',contour=[[100,30],[120,30],[120,50],[100,50]])}}
    rows=list(generate_pairs(objects,{'map':[0,0,200,200]},['map']))
    assert len(rows)==2
    for row in rows:
        target=np.asarray(row['target_xy'])*200
        assert Polygon(objects['map'][row['hidden_target_id']]['contour']).covers(Point(target))
        assert row['signed_anchor_boundary_distance_m']>=5


def test_perfect_inverse_predictions_cancel_correct_area_centers_not_vertex_means():
    import torch
    from types import SimpleNamespace
    from graphnav.bidirectional_train import batch_loss
    grid=torch.tensor([[[.25,.25],[.75,.25],[.25,.75],[.75,.75]]]*2)
    batch={key:torch.zeros(2,1) for key in ('node_types','node_valid','entity_valid','adjacency','edges','prior','prior_buildings','rules','relation_fields','pair_geometry')}
    batch.update(nodes=torch.tensor([[[.20,.20]],[[.65,.20]]]),grid_xy=grid,
        text=['east of a','west of b'],node_names=[['a'],['b']],map_sizes=torch.full((2,2),400.),
        target_xy=torch.tensor([[.75,.25],[.25,.25]]),anchor_center_label=torch.tensor([[.25,.25],[.75,.25]]))
    class Encoder:
        def encode_tokens(self,text):return torch.zeros(2,1,768),torch.ones(2,1,dtype=torch.bool)
        def encode(self,names):return torch.zeros(2,768)
    class Model:
        def __call__(self,inputs,**kwargs):
            assert 'target_xy' not in inputs and 'anchor_center_label' not in inputs
            return SimpleNamespace(belief_logits=torch.tensor([[-1000.,1000.,-1000.,-1000.],[1000.,-1000.,-1000.,-1000.]]))
    _,parts,_=batch_loss(Model(),batch,Encoder(),torch.device('cpu'))
    assert parts['inverse']==0.
