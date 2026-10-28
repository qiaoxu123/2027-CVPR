import pytest
import torch
from experiments.a2_waldo_landmark_belief.relation_v3_model import RelationV3Network


def inputs():
    return dict(instruction_tokens=torch.randn(2,4,8),token_valid=torch.ones(2,4,dtype=torch.bool),
        mention_mask=torch.ones(2,2,4,dtype=torch.bool),landmark_name_tokens=torch.randn(2,2,3,8),name_valid=torch.ones(2,2,3,dtype=torch.bool),
        link_valid=torch.ones(2,2,dtype=torch.bool),box_rgb=torch.randn(2,3,6),box_valid=torch.ones(2,3,dtype=torch.bool),
        pair_geometry=torch.randn(2,3,2,14),pair_valid=torch.ones(2,3,2,dtype=torch.bool),offset_limit=torch.ones(2,3,2),box_kind=torch.zeros(2,3,dtype=torch.long))


def test_external_ranking_and_match_are_independent_of_A0_prior():
    torch.manual_seed(0);model=RelationV3Network(text_dim=8,vision_dim=6,hidden=8,heads=2).eval();x=inputs()
    a=model(**x,a0_context=torch.zeros(2,3,6));b=model(**x,a0_context=torch.full((2,3,6),1000.))
    assert torch.equal(a['scores'],b['scores']) and torch.equal(a['candidate_match_logits'],b['candidate_match_logits'])
    assert not a['offset'].any() and not any(p.requires_grad for p in model.base.offset.parameters())
    with pytest.raises(TypeError):model(**x,target_xy=torch.zeros(2,2))


def test_public_risk_head_has_gradients_without_backpropagating_into_actor_features():
    torch.manual_seed(0);model=RelationV3Network(text_dim=8,vision_dim=6,hidden=8,heads=2)
    features=torch.randn(4,15,requires_grad=True);model.predict_risk(features).square().mean().backward()
    assert features.grad is None and model.risk[-1].weight.grad.norm()>0


def test_warmstart_copies_only_evidence_model_and_category_not_prior_shortcuts():
    from experiments.a2_waldo_landmark_belief.revision_model import RevisionNetwork
    torch.manual_seed(0);old=RevisionNetwork(text_dim=8,vision_dim=6,hidden=8,heads=2)
    model=RelationV3Network(text_dim=8,vision_dim=6,hidden=8,heads=2)
    copied=model.warmstart(dict(model=old.state_dict(),model_config=old.config))
    assert copied and torch.equal(model.base.scorer.box[0].weight,old.base.scorer.box[0].weight)
    assert not hasattr(model,'rank_context') and not hasattr(model,'trust_context')
