import numpy as np
import pytest
from graphnav.complete_landmark_rgb import tile_plan


def test_local_crops_use_only_public_polygon_and_whole_extent():
    poly=[[[0,0],[200,0],[200,20],[0,20]]]
    plan=tile_plan(poly)
    assert plan['whole']['polygon_fraction']>.999
    assert len(plan['tiles'])>1
    full=np.asarray(plan['whole']['footprint_world']);lo=full.min(0);hi=full.max(0)
    for tile in plan['tiles']:
        fp=np.asarray(tile['footprint_world']);assert (fp>=lo-1e-4).all() and (fp<=hi+1e-4).all()
        assert tile['side_m']<=tile['scale_m']+1e-4
    with pytest.raises(TypeError):tile_plan(poly,target_xy=[1,1])


def test_duplicate_patch_evidence_is_averaged_and_zero_joint_matches_legacy():
    import torch
    from types import SimpleNamespace
    from graphnav.complete_landmark_rgb import refine_cached
    model=SimpleNamespace(pgvb=lambda **x:x['image_features'],joint_logits=lambda x:torch.zeros(*x.shape[:2],1))
    b=torch.zeros(1,9);ctx=torch.randn(1,9,128);obs=dict(image_features=torch.tensor([[1.,-1.]]),valid=torch.ones(1,2,dtype=torch.bool),local_uv=torch.tensor([[[.2,.2],[.8,.8]]]))
    out,_=refine_cached(model,b,ctx,obs)
    both={k:torch.cat([v,v],1) for k,v in obs.items()}
    dup,_=refine_cached(model,b,ctx,both)
    joint,_=refine_cached(model,b,ctx,obs,joint=True)
    torch.testing.assert_close(out,dup);torch.testing.assert_close(out,joint)
    assert out.abs().max()<=.5


def test_multiview_gradient_controls_and_privileged_checkpoint_rejected():
    import torch
    from types import SimpleNamespace
    from graphnav.complete_landmark_rgb import refine_views
    from graphnav.polygon_guided_visual import PolygonGuidedVisualBelief
    from graphnav.full_relation_training import load_full_relation_model
    torch.manual_seed(42)
    model=SimpleNamespace(pgvb=PolygonGuidedVisualBelief(dimension=8,hidden=8,heads=2),joint_logits=torch.nn.Linear(130,1))
    p=dict(image_features=torch.randn(2,4,3,8),name_text=torch.randn(2,8),type_text=torch.randn(2,8),polygon_tokens=torch.randn(2,16,4),valid=torch.tensor([[True,True,True,False]]*2),local_uv=torch.tensor([[[.1,.1],[.5,.5],[.9,.9],[.3,.2]]]*2))
    base=torch.randn(2,9);dense=torch.randn(2,9,128)
    out,covered=refine_views(model,base,dense,[p,None],joint=True)
    assert covered[0].any() and not covered[1].any()
    torch.testing.assert_close(out[1],base[1]);assert (out-base).abs().max()<=.5
    out.square().sum().backward()
    assert sum(float(x.grad.norm()) for x in model.pgvb.parameters() if x.grad is not None)>0
    assert model.joint_logits.weight.grad.norm()>0
    before=p['image_features'].clone()
    for c in ('zero','shuffle'):assert torch.isfinite(refine_views(model,base,dense,[p,None],control=c)[0]).all()
    torch.testing.assert_close(before,p['image_features'])
    with pytest.raises(ValueError,match='Unsupported'):load_full_relation_model(dict(format='unified-pgvb-privileged-visual-train-v1'),device='cpu')


def test_polygon_boundary_touch_has_no_area_and_uses_shape_fallback():
    from graphnav.complete_landmark_rgb import registered_shape_tokens
    from graphnav.polygon_guided_visual import polygon_shape_tokens
    from graphnav.local_landmark_anchor import local_anchor_tokens
    poly=[[[0,0],[1,0],[1,1],[0,1]]]
    touch=np.array([[1,0],[2,0],[2,1],[1,1]],float)
    actual=registered_shape_tokens(poly,touch)
    np.testing.assert_allclose(actual,polygon_shape_tokens(poly,touch))
    overlap=np.array([[.5,0],[1.5,0],[1.5,1],[.5,1]],float)
    np.testing.assert_allclose(registered_shape_tokens(poly,overlap),local_anchor_tokens(poly,overlap))


def test_vectorized_sparse_regions_exactly_preserve_native_masks_and_quality():
    from graphnav.complete_landmark_rgb import sparse_native_regions
    from graphnav.native_dense_regions import native_dense_regions,sparse_view
    rng=np.random.default_rng(42)
    for angle in (0.,.5,1.57):
        xy=np.array([[-50,-50],[50,-50],[50,50],[-50,50]],float)
        rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        obs=dict(bgr=rng.integers(0,256,(384,384,3),dtype=np.uint8),depth_valid=rng.random((384,384))>.05,footprint_world=xy@rot+np.array([100,200]))
        old=sparse_view(native_dense_regions(obs));new=sparse_native_regions(obs)
        for key in old:np.testing.assert_array_equal(old[key],new[key],err_msg=key)


def test_complete_same_name_parts_never_uses_nearest_target_or_other_types():
    from scripts.prepare_complete_landmark_rgb import complete_public_parts
    original=[dict(id='1',name='Example Road',object_type='TrafficRoad',contour=[[0,0],[.2,0],[.2,.2],[0,.2]])]
    objects={'1':dict(name='Example Road',object_type='TrafficRoad',contour=[[0,0],[2,0],[2,2],[0,2]]),'2':dict(name='Example Road',object_type='TrafficRoad',contour=[[8,8],[9,8],[9,9],[8,9]]),'3':dict(name='Example Road',object_type='Building',contour=[[0,0],[2,0],[2,2],[0,2]])}
    out=complete_public_parts(original,objects,[0,0,10,10])
    assert [x['id'] for x in out]==['1','2']
    assert out[0]==original[0]
    with pytest.raises(TypeError):complete_public_parts(original,objects,[0,0,10,10],target_xy=[0,0])
