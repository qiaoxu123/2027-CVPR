import torch
from tests.test_a2_revision_model_data import inputs
from experiments.a2_waldo_landmark_belief.revision_model import RevisionNetwork
from experiments.a2_waldo_landmark_belief.revision_loss import revision_loss


def batch():
    return dict(network=inputs(),spatial=dict(
      base_logits=torch.tensor([[.4,.3,.2,.1],[.4,.3,.2,.1]]).log(),
      grid_world=torch.tensor([[[0.,0.],[10.,0.],[20.,0.],[30.,0.]]]*2),
      candidate_xy=torch.tensor([[[0.,0.],[30.,0.],[20.,0.]]]*2),
      candidate_support=torch.tensor([[[1,0,0,0],[0,0,0,1],[0,0,1,0]]]*2,dtype=torch.bool)),
      labels=dict(ranking=torch.tensor([[1.,0,0],[0,1.,0]]),rank_label_mask=torch.ones(2,3,dtype=torch.bool),
       offset_valid=torch.tensor([[1,0,0],[0,1,0]],dtype=torch.bool),angle=torch.tensor([[[1.,0.],[1.,0.]]]*2),
       angle_valid=torch.ones(2,2,dtype=torch.bool),distance=torch.ones(2,2),offset=torch.zeros(2,3,2),
       target_xy=torch.tensor([[0.,0.],[30.,0.]]),target_kind=torch.tensor([0,1])))


def test_spatial_and_benefit_losses_backpropagate_without_private_forward_inputs():
    torch.manual_seed(0);m=RevisionNetwork(text_dim=12,vision_dim=16,hidden=16,heads=2);b=batch()
    output=m(**b['network']);loss,stats=revision_loss(output,b,refine=True)
    assert torch.isfinite(loss) and stats['proposal_covered_views']==2
    loss.backward()
    for prefix in ('base.scorer.rank_head.','category.','benefit.'):
      assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for n,p in m.named_parameters() if n.startswith(prefix))


def test_uncovered_targets_skip_pure_proposal_ce_but_map_loss_remains_finite():
    m=RevisionNetwork(text_dim=12,vision_dim=16,hidden=16,heads=2);b=batch();b['labels']['target_xy'].fill_(200.)
    loss,stats=revision_loss(m(**b['network']),b,refine=True)
    assert torch.isfinite(loss) and stats['proposal_covered_views']==0 and stats['proposal_localization']==0
    loss.backward()


def test_no_supported_candidates_fallback_has_finite_loss_and_no_fake_coverage():
    m=RevisionNetwork(text_dim=12,vision_dim=16,hidden=16,heads=2);b=batch();b['spatial']['candidate_support'].zero_()
    loss,stats=revision_loss(m(**b['network']),b,refine=True)
    assert torch.isfinite(loss) and stats['proposal_covered_views']==0
    assert stats['corrected']==0 and stats['broken']==0
