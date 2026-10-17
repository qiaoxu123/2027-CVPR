import copy
import numpy as np
import pytest
import torch

from graphnav.polygon_ablation import pair_features, geometry_view, representation_fields, node_features
from graphnav.candidate_geometry import relative_entity_features
from graphnav.semantic_graph import prepare_semantic_graph
from graphnav.geometry import grid_centers
from graphnav.full_localization import FullTokenGeometryLocalizer


def entity(poly, kind='building'):
    return dict(name='Anchor',kind=kind,geometry_m=np.asarray(poly,np.float32))


@pytest.mark.parametrize('rep',['centroid','bounding_box','full_polygon'])
def test_ten_dimensional_interface_and_shared_center(rep):
    e=entity([[0,0],[4,0],[4,2],[0,2]])
    points=np.array([[2,1],[5,1]],np.float32)
    features,mask=pair_features(points,[e],3,rep)
    assert features.shape==(2,3,10) and mask.shape==(3,10)
    assert np.isfinite(features).all() and not features[:,1:].any()
    np.testing.assert_allclose(features[:,:1,:5],relative_entity_features(points,[e],3)[:,:1,:5])
    if rep=='centroid':
        assert not features[...,5:9].any() and not mask[0,5:9].any()
    else:
        assert features[0,0,5]<0 and features[0,0,6]==1
        assert features[1,0,5]>0 and features[1,0,6]==0


def test_open_road_not_closed_and_road_surface_is_area():
    line=[[0,0],[4,0],[4,4]]
    f,_=pair_features([[3,1]],[entity(line,'road')],1,'full_polygon')
    assert f[0,0,6]==0 and f[0,0,5]>0 and f[0,0,9]==1
    for rep in ('centroid','bounding_box'):
        f,m=pair_features([[3,1]],[entity(line,'road')],1,rep)
        assert f[0,0,5]==f[0,0,6]==0 and m[0,5]==m[0,6]==0
    f,_=pair_features([[3,1]],[entity(line,'road_surface')],1,'full_polygon')
    assert f[0,0,6]==1 and f[0,0,9]==0


def test_inside_removal_does_not_silently_claim_signed_distance_removed():
    e=entity([[0,0],[4,0],[4,4],[0,4]])
    f,_=pair_features([[2,2]],[e],1,'no_inside')
    assert f[0,0,6]==0 and f[0,0,5]<0
    f,_=pair_features([[2,2]],[e],1,'no_inside_information')
    assert f[0,0,6]==0 and f[0,0,5]>0


def item_for(poly):
    bounds=[-4,-4,4,4]
    raw=dict(name='Anchor',id='x',object_type='Building',contour=((np.asarray(poly)+4)/8).tolist())
    g=prepare_semantic_graph([raw],[],'near Anchor',bounds,max_landmarks=1,max_roads=0,max_relations=2)
    grid=grid_centers(4)
    pts=-4+grid.numpy()*8
    from graphnav.semantic_geometry import prior_and_rule_fields
    p,r,f=prior_and_rule_fields(pts,g['entities'],g['relations'])
    rel=np.zeros((16,2),np.float32);rel[:,:f.shape[1]]=f
    item=dict(nodes=torch.from_numpy(g['nodes']),node_types=torch.from_numpy(g['node_types']),node_valid=torch.from_numpy(g['node_mask']),
              entity_valid=torch.tensor([True,False,False]),edges=torch.from_numpy(g['edges']),adjacency=torch.from_numpy(g['adjacency']),
              prior=torch.from_numpy(p),prior_buildings=torch.from_numpy(p),rules=torch.from_numpy(r),relation_fields=torch.from_numpy(rel),
              grid_xy=grid,pair_geometry=torch.from_numpy(relative_entity_features(pts,g['entities'],3).astype(np.float16)),target_xy=torch.tensor([.1,.9]))
    return item,g,bounds


@pytest.mark.parametrize('rep',['centroid','bounding_box'])
@pytest.mark.parametrize('polygon_b',[
    [[-2,0],[0,-2],[2,0],[0,2]],
    [[-2,-2],[0,-2],[2,0],[2,2],[0,2],[-2,0]],
])
def test_end_to_end_has_no_hidden_shape_information(rep,polygon_b):
    a,g,b=item_for([[-2,-2],[2,-2],[2,2],[-2,2]])
    c,h,_=item_for(polygon_b)
    first=geometry_view(a,g,b,representation=rep,scope='end_to_end')
    second=geometry_view(c,h,b,representation=rep,scope='end_to_end')
    for key in ('nodes','prior','prior_buildings','rules','relation_fields','pair_geometry','adjacency','edges'):
        torch.testing.assert_close(first[key],second[key],msg=key)
    descriptor=geometry_view(a,g,b,representation='centroid',scope='descriptor_only')
    assert descriptor['nodes'][0,5]>0, 'Descriptor Only intentionally retains Polygon node-area bypass'


@pytest.mark.parametrize('scope',['descriptor_only','end_to_end'])
def test_full_polygon_preserves_archived_inputs_and_labels_cannot_change_geometry(scope):
    item,g,b=item_for([[-2,-2],[2,-2],[2,2],[-2,2]])
    out=geometry_view(item,g,b,scope=scope)
    for key in ('nodes','prior','prior_buildings','rules','relation_fields','pair_geometry'):
        torch.testing.assert_close(out[key],item[key],rtol=0,atol=0)
    item['target_xy']=torch.tensor([123.,-999.])
    changed=geometry_view(item,g,b,scope=scope)
    for key in ('nodes','prior','rules','relation_fields','pair_geometry'):
        torch.testing.assert_close(out[key],changed[key],rtol=0,atol=0)


def test_geometry_attention_never_calls_gat_message_passing_and_capacity_matches():
    item,g,b=item_for([[-2,-2],[2,-2],[2,2],[-2,2]])
    model=FullTokenGeometryLocalizer(hidden=16,heads=2,max_relations=2).eval()
    def forbidden(*args,**kwargs):
        raise AssertionError('GAT message passing must not run')
    model.gnn.layer1.forward=forbidden;model.gnn.layer2.forward=forbidden
    count=sum(p.numel() for p in model.parameters())
    for rep in ('centroid','bounding_box','full_polygon'):
        view=geometry_view(item,g,b,representation=rep,scope='end_to_end')
        batch={k:v[None] for k,v in view.items() if isinstance(v,torch.Tensor)}
        f=model(batch,variant='geometry',tokens=torch.zeros(1,3,768),token_mask=torch.ones(1,3,dtype=torch.bool),node_emb=torch.zeros(1,3,768))
        assert torch.isfinite(f.belief_logits).all()
        assert sum(p.numel() for p in model.parameters())==count


def test_representation_views_do_not_mutate_shared_public_feature_cache():
    item,g,b=item_for([[-2,-2],[2,-2],[2,2],[-2,2]])
    original={k:v.clone() for k,v in item.items() if isinstance(v,torch.Tensor)}
    geometry=[e['geometry_m'].copy() for e in g['entities']]
    for scope in ('descriptor_only','end_to_end'):
        for rep in ('centroid','bounding_box','full_polygon'):
            geometry_view(item,g,b,representation=rep,scope=scope)
            for key,value in original.items():torch.testing.assert_close(item[key],value,rtol=0,atol=0)
            for e,xy in zip(g['entities'],geometry):np.testing.assert_array_equal(e['geometry_m'],xy)


def test_public_query_cache_matches_uncached_conversion_exactly(tmp_path,monkeypatch):
    import json
    import scripts.polygon_ablation_navigation as nav
    from graphnav.checkpointing import file_sha256
    plan={'args':dict(max_landmarks=16,max_roads=0,max_relations=8,prior_sigma_m=25.,rule_sigma_m=25.)}
    (tmp_path/'frozen_plan.json').write_text(json.dumps(plan))
    record=dict(map_name='synthetic',instruction='near Anchor',world_bounds=[-4,-4,4,4],physical_map_sizes=[8,8],
                landmarks=[dict(name='Anchor',object_type='Building',contour=[[.25,.25],[.75,.25],[.75,.75],[.25,.75]])],target_xy=[0.,0.])
    key=json.dumps(record,sort_keys=True);raw=tmp_path/'raw.jsonl';raw.write_text(key+'\n')
    direct=nav.make_dataset(plan,raw)[0];calls=[];original=nav.make_dataset
    def counted(*args):calls.append(1);return original(*args)
    monkeypatch.setattr(nav,'OUT',tmp_path);monkeypatch.setattr(nav,'make_dataset',counted)
    nav.public_query_features.cache_clear()
    try:
        cached,_=nav.public_query_features(key,file_sha256(tmp_path/'frozen_plan.json'))
        repeated,_=nav.public_query_features(key,file_sha256(tmp_path/'frozen_plan.json'))
        assert repeated is cached and len(calls)==1
        for k,v in direct.items():
            if isinstance(v,torch.Tensor):torch.testing.assert_close(cached[k],v,rtol=0,atol=0)
            else:assert cached[k]==v
    finally:nav.public_query_features.cache_clear()
