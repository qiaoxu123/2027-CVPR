import torch
from tests.test_a2_relation_v3_model import inputs
from experiments.a2_waldo_landmark_belief.relation_v3_model import RelationV3Network
from experiments.a2_waldo_landmark_belief.relation_v3_loss import relation_v3_loss


def batch():
    return dict(network=inputs(),spatial=dict(
        base_logits=torch.tensor([[.8,.1,.1],[.1,.1,.8]]).log(),
        grid_world=torch.tensor([[[0.,0.],[50.,0.],[100.,0.]]]*2),
        candidate_xy=torch.tensor([[[0.,0.],[50.,0.],[100.,0.]]]*2),
        candidate_support=torch.eye(3,dtype=torch.bool)[None].expand(2,-1,-1)),
        labels=dict(target_xy=torch.zeros(2,2),target_kind=torch.tensor([0,0]),
            angle=torch.tensor([[[1.,0.],[1.,0.]]]*2),angle_valid=torch.ones(2,2,dtype=torch.bool),distance=torch.ones(2,2),
            localization_utility=torch.tensor([[1,0,0],[1,0,0]],dtype=torch.bool),utility_valid=torch.ones(2,3,dtype=torch.bool),
            identity_label_mask=torch.zeros(2,3,dtype=torch.bool),sample_weight=torch.ones(2)))


def test_full_pool_spatial_and_refinement_losses_train_actor_and_separate_risk_head():
    torch.manual_seed(0);m=RelationV3Network(text_dim=8,vision_dim=6,hidden=8,heads=2);b=batch()
    loss,stats=relation_v3_loss(m,m(**b['network']),b,refine=True)
    assert torch.isfinite(loss) and stats['rank_views']==2 and stats['utility_candidates']==6
    loss.backward()
    for prefix in ('base.scorer.rank_head.','base.match.','category.','risk.'):
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for n,p in m.named_parameters() if n.startswith(prefix))
    assert all(p.grad is None for p in m.base.offset.parameters())


def test_per_row_source_weights_have_absolute_effect_and_identity_is_unused():
    torch.manual_seed(0);m=RelationV3Network(text_dim=8,vision_dim=6,hidden=8,heads=2);b=batch();out=m(**b['network'])
    a,_=relation_v3_loss(m,out,b,refine=True)
    b['labels']['sample_weight'].fill_(8);b['labels']['identity_label_mask'].fill_(True)
    c,_=relation_v3_loss(m,out,b,refine=True)
    assert torch.allclose(c,8*a)


def test_no_candidate_support_falls_back_exactly_without_nan_gradients():
    torch.manual_seed(0);m=RelationV3Network(text_dim=8,vision_dim=6,hidden=8,heads=2);b=batch()
    b['spatial']['candidate_support'].zero_();b['labels']['utility_valid'].zero_();b['labels']['localization_utility'].zero_()
    loss,stats=relation_v3_loss(m,m(**b['network']),b,refine=True)
    assert torch.isfinite(loss) and stats['proposal_available_views']==0
    assert stats['corrected']==stats['broken']==0
    loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())


def test_stage_one_does_not_train_risk_head_and_all_unknown_supervision_is_finite():
    torch.manual_seed(0);m=RelationV3Network(text_dim=8,vision_dim=6,hidden=8,heads=2);b=batch()
    b['labels']['target_kind'].fill_(-1);b['labels']['angle_valid'].zero_();b['labels']['utility_valid'].zero_()
    loss,stats=relation_v3_loss(m,m(**b['network']),b,refine=False);loss.backward()
    assert torch.isfinite(loss) and stats['ranking']==0
    assert all(p.grad is None for p in m.risk.parameters())


def test_risk_only_loss_cannot_change_actor_even_when_public_features_depend_on_actor():
    from experiments.a2_waldo_landmark_belief.relation_v3_fusion import proposal_with_quality
    from experiments.a2_waldo_landmark_belief.risk_probe import public_risk_features
    torch.manual_seed(0);m=RelationV3Network(text_dim=8,vision_dim=6,hidden=8,heads=2);b=batch();out=m(**b['network'])
    q,available,quality=proposal_with_quality(b['spatial'],out['scores'],out['candidate_match_logits'],out['reliable'])
    features=public_risk_features(b['spatial']['base_logits'],q,b['spatial']['grid_world'],quality,available)
    assert features.requires_grad
    torch.nn.functional.cross_entropy(m.predict_risk(features),torch.tensor([1,2])).backward()
    assert all(p.grad is None for n,p in m.named_parameters() if not n.startswith('risk.'))
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.risk.parameters())


def test_spatially_unsuitable_unknown_identity_candidate_is_in_rank_denominator():
    torch.manual_seed(0);m=RelationV3Network(text_dim=8,vision_dim=6,hidden=8,heads=2);b=batch();out=m(**b['network'])
    out['scores'].retain_grad();loss,_=relation_v3_loss(m,out,b)
    loss.backward()
    # Candidate2 is a spatial negative with unknown identity, not excluded from
    # localization utility supervision. Reducing its score reduces this loss.
    assert not b['labels']['identity_label_mask'].any()
    assert (out['scores'].grad[:,2]>0).all()
