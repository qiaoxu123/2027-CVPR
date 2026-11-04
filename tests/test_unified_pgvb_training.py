import torch
from graphnav.unified_pgvb import UnifiedPGVB
from graphnav.unified_pgvb_training import configure_visual_training,weak_landmark_loss,frame_order

def test_only_unified_visual_head_has_gradients():
 model=UnifiedPGVB();names=configure_visual_training(model)
 assert names and all(k.startswith('pgvb.') for k in names)
 assert not model.joint_ready and not model.direction_ready and not model.target_ready
 batch=dict(image_features=torch.randn(2,8,3,1024),name_text=torch.randn(2,1024),type_text=torch.randn(2,1024),polygon_tokens=torch.randn(2,16,4),valid=torch.ones(2,8,dtype=torch.bool))
 score=model(**batch)['landmark_scores'];labels=torch.tensor([[1,1,0,0,-1,-1,0,0],[0,0,1,1,-1,-1,0,0]])
 loss=weak_landmark_loss(score,labels,torch.tensor([0,0]),torch.tensor([0,0]))['total'];loss.backward()
 assert all(p.grad is None for k,p in model.named_parameters() if not k.startswith('pgvb.'))
 assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.pgvb.parameters())

def test_unknown_regions_never_negative_and_positive_only_is_valid():
 scores=torch.tensor([[1.,2.,999.,-999.]],requires_grad=True);lab=torch.tensor([[1,1,-1,-1]])
 loss=weak_landmark_loss(scores,lab,torch.tensor([0]),torch.tensor([0]));loss['total'].backward()
 assert torch.isfinite(loss['total']);assert torch.equal(scores.grad[0,2:],torch.zeros(2))

def test_frame_order_supports_exact_mid_epoch_resume():
 full=frame_order([1,3,8,9,10],42,3);assert full==frame_order([1,3,8,9,10],42,3)
 assert full[:2]+frame_order([1,3,8,9,10],42,3)[2:]==full
 assert set(full)=={1,3,8,9,10}

def test_overlapping_public_regions_are_not_false_hard_negatives():
 labels=torch.tensor([[1,1,0,-1],[1,1,0,-1]])
 score=torch.zeros(2,4,requires_grad=True)
 result=weak_landmark_loss(score,labels,torch.tensor([9,9]),torch.tensor([0,0]))
 assert result['pairs']==0
 disjoint=torch.tensor([[1,1,0,0],[0,0,1,1]])
 result=weak_landmark_loss(score,disjoint,torch.tensor([9,9]),torch.tensor([0,0]))
 assert result['pairs']==2

def test_visual_training_does_not_run_static_geometry():
 from torch import nn
 class FrozenGeometry(nn.Module):
  hidden=128
  def forward(self,*args,**kwargs):raise AssertionError('Static Geometry must not be recomputed for local loss')
 model=UnifiedPGVB(geometry=FrozenGeometry());configure_visual_training(model)
 batch=dict(image_features=torch.randn(1,8,3,1024),name_text=torch.randn(1,1024),type_text=torch.randn(1,1024),polygon_tokens=torch.randn(1,16,4),valid=torch.ones(1,8,dtype=torch.bool))
 assert model(**batch)['landmark_scores'].shape==(1,8)
