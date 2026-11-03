import inspect
import pytest
import torch

from graphnav.local_chain_position import LocalChainPositionHead, position_loss


def inputs(valid=True):
    torch.manual_seed(3)
    return dict(prior_logits=torch.randn(2,16), local_uv=torch.rand(2,5,2),
                image_features=torch.randn(2,5,3,12), target_text=torch.randn(2,12),
                context_text=torch.randn(2,12), public_geometry=torch.randn(2,5,31),
                quality=torch.rand(2,5,6), local_valid=torch.full((2,5),valid,dtype=torch.bool))


def test_unknown_observation_preserves_original_distribution():
    model=LocalChainPositionHead(dimension=12,hidden=8)
    x=inputs(False); out=model(**x)
    torch.testing.assert_close(out[:,:16].softmax(-1),x['prior_logits'].softmax(-1))
    torch.testing.assert_close(out.log_softmax(-1)[:,:16],x['prior_logits'].log_softmax(-1))
    assert torch.isneginf(out[:,16:]).all()


def test_invalid_visual_values_cannot_change_position_distribution():
    model=LocalChainPositionHead(dimension=12,hidden=8)
    x=inputs();x['local_valid'][:,2:]=False
    a=model(**x)
    x['image_features'][:,2:]=1000
    b=model(**x)
    torch.testing.assert_close(a,b)


def test_position_supervision_backpropagates_to_visual_and_geometry():
    model=LocalChainPositionHead(dimension=12,hidden=8)
    x=inputs();logits=model(**x)
    world=torch.arange(21).float()[None,:,None].expand(2,-1,2)
    target=torch.tensor([[19.,19.],[20.,20.]])
    loss=position_loss(logits,world,target,global_cells=16,radius_m=2.)
    loss['total'].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.image.weight.grad.norm()>0
    assert model.geometry.weight.grad.norm()>0


def test_absent_target_in_local_candidates_does_not_make_false_positive_labels():
    model=LocalChainPositionHead(dimension=12,hidden=8)
    logits=model(**inputs())
    world=torch.cat((torch.zeros(2,16,2),torch.full((2,5,2),100.)),1)
    loss=position_loss(logits,world,torch.zeros(2,2),global_cells=16)
    assert loss['local_supervised']==0
    assert loss['local']==0
    assert torch.isfinite(loss['total'])


def test_model_forward_has_no_private_label_interface():
    params=set(inspect.signature(LocalChainPositionHead.forward).parameters)
    assert not params & {'target_world','target_id','target_contour','target_bbox','labels','future_frames'}
    model=LocalChainPositionHead(dimension=12,hidden=8)
    with pytest.raises(TypeError):
        model(**inputs(),target_world=torch.zeros(2,2))
