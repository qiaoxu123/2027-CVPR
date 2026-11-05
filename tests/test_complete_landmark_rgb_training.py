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
