import torch
from graphnav.geometry_guided_visual import LandmarkVisualBelief,visual_world_beliefs,landmark_visual_loss


def test_visual_scores_take_no_coordinates_and_have_finite_image_gradient():
    torch.manual_seed(0);model=LandmarkVisualBelief(dimension=8,hidden=4)
    image=torch.randn(2,4,3,8,requires_grad=True);valid=torch.ones(2,4,dtype=torch.bool)
    text=torch.randn(2,8)
    scores=model(image_features=image,name_text=text,type_text=text,valid=valid)
    scores.square().mean().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all() and image.grad.abs().sum()>0
    try:model(image_features=image,name_text=text,type_text=text,valid=valid,polygon=torch.zeros(2,4))
    except TypeError:pass
    else:raise AssertionError('Geometry coordinates must not enter visual scorer')


def test_no_public_projection_support_preserves_geometry_exactly():
    geometry=torch.randn(2,784);uv=torch.rand(2,4,2);scores=torch.randn(2,4)
    result=visual_world_beliefs(geometry,uv,scores,torch.ones(2,4,dtype=torch.bool),torch.zeros(2,dtype=torch.bool))
    assert torch.equal(result['fused'],geometry.log_softmax(-1))
    empty=visual_world_beliefs(geometry,uv,scores,torch.zeros(2,4,dtype=torch.bool),torch.zeros(2,dtype=torch.bool))
    assert not empty['has_view'].any()


def test_contrast_ignores_single_view_groups_without_nan_gradients():
    model=LandmarkVisualBelief(dimension=8,hidden=4)
    batch=dict(image_features=torch.randn(4,4,3,8),name_text=torch.randn(4,8),type_text=torch.randn(4,8),
        valid=torch.ones(4,4,dtype=torch.bool),labels=torch.tensor([[1,0,-1,0]]*4),
        geometry_logits=torch.randn(4,784),local_uv=torch.rand(4,4,2),
        group_ids=torch.tensor([0,0,1,2]),view_ids=torch.tensor([1,2,3,4]))
    loss=landmark_visual_loss(model,batch)
    assert loss['contrast_rows']==2
    loss['total'].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    batch['view_ids']=torch.tensor([1,1,3,4])
    assert landmark_visual_loss(model,batch)['contrast_rows']==0


def test_controller_bridge_keeps_zero_evidence_exact_and_preserves_raw_logit_gauge():
    from graphnav.geometry_guided_runtime import fused_controller_field
    from graphnav.full_localization import GoalField
    uv=torch.rand(1,4,2);dense=torch.randn(1,4,6);logits=torch.tensor([[21.,19.,18.,17.]])
    base=GoalField(logits,torch.tensor([[0,1]]),uv[:,:2],dense[:,:2],dense_context=dense)
    assert fused_controller_field(base,uv,torch.zeros(1,4),True) is base
    assert fused_controller_field(base,uv,torch.ones(1,4),False) is base
    result=fused_controller_field(base,uv,torch.tensor([[0.,4.,0.,0.]]),True)
    assert torch.equal(result.belief_logits,torch.tensor([[21.,23.,18.,17.]]))
    assert int(result.candidate_ids[0,0])==1
