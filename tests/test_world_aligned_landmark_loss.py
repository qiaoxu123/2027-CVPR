import torch
from graphnav.geometry_guided_visual import LandmarkVisualBelief
from graphnav.world_aligned_landmark_loss import world_landmark_visual_loss


def test_world_grid_distillation_backpropagates_without_unobserved_nan():
    model=LandmarkVisualBelief(dimension=8,hidden=4)
    batch=dict(image_features=torch.randn(4,6,3,8,requires_grad=True),name_text=torch.randn(4,8),type_text=torch.randn(4,8),
        valid=torch.ones(4,6,dtype=torch.bool),labels=torch.tensor([[1,0,-1,0,0,1]]*4),
        geometry_logits=torch.randn(4,784),local_uv=torch.rand(4,6,2)*.3+.3,
        projection_supported=torch.tensor([True,True,True,False]),
        group_ids=torch.tensor([0,0,1,2]),view_ids=torch.tensor([1,2,3,4]))
    loss=world_landmark_visual_loss(model,batch)
    assert torch.isfinite(loss['total'])
    loss['total'].backward()
    assert torch.isfinite(batch['image_features'].grad).all()
    assert batch['image_features'].grad.abs().sum()>0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
