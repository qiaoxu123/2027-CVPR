import torch
from experiments.a2_waldo_landmark_belief.model import LandmarkBeliefNetwork, supervision_loss


def sample():
    torch.manual_seed(4)
    return dict(instruction_tokens=torch.randn(2,7,12), token_valid=torch.ones(2,7,dtype=torch.bool),
                mention_mask=torch.ones(2,3,7,dtype=torch.bool), landmark_name_tokens=torch.randn(2,3,4,12),
                name_valid=torch.ones(2,3,4,dtype=torch.bool), link_valid=torch.ones(2,3,dtype=torch.bool),
                box_rgb=torch.randn(2,5,16), box_valid=torch.ones(2,5,dtype=torch.bool),
                pair_geometry=torch.randn(2,5,3,14), pair_valid=torch.ones(2,5,3,dtype=torch.bool),
                offset_limit=torch.ones(2,5,2)*4)


def test_only_new_network_receives_finite_gradients_with_padded_candidates():
    net=LandmarkBeliefNetwork(text_dim=12,vision_dim=16,hidden=16,heads=2)
    inputs=sample();inputs['box_valid'][1,3:]=False
    out=net(**inputs)
    labels=dict(ranking=torch.tensor([[1.,0,0,0,0],[0,1.,0,0,0]]),
                rank_label_mask=torch.tensor([[True,True,False,False,False],[True,True,True,False,False]]),
                angle=torch.randn(2,3,2),angle_valid=torch.ones(2,3,dtype=torch.bool),
                distance=torch.rand(2,3),offset=torch.zeros(2,5,2),offset_valid=torch.zeros(2,5,dtype=torch.bool),
                match=torch.ones(2),match_valid=torch.ones(2,dtype=torch.bool))
    loss,stats=supervision_loss(out,labels,refine=False)
    loss.backward()
    assert stats['rank_views']==2 and torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    assert out['reliable'][1].tolist()==[True,True,True,False,False]
    assert (out['offset'].abs()<=inputs['offset_limit']).all()


def test_unknown_boxes_are_not_negative_ranking_examples():
    net=LandmarkBeliefNetwork(text_dim=12,vision_dim=16,hidden=16,heads=2)
    out=net(**sample()); out['scores'].retain_grad()
    labels=dict(ranking=torch.tensor([[1.,0,0,0,0],[1.,0,0,0,0]]),
                rank_label_mask=torch.tensor([[True,True,False,False,False]]*2),
                angle=torch.zeros(2,3,2),angle_valid=torch.zeros(2,3,dtype=torch.bool),distance=torch.zeros(2,3),
                offset=torch.zeros(2,5,2),offset_valid=torch.zeros(2,5,dtype=torch.bool),
                match=torch.zeros(2),match_valid=torch.zeros(2,dtype=torch.bool))
    loss,_=supervision_loss(out,labels,refine=False);loss.backward()
    assert (out['scores'].grad[:,2:]==0).all()


def test_missing_landmarks_force_ineligible_output():
    net=LandmarkBeliefNetwork(text_dim=12,vision_dim=16,hidden=16,heads=2)
    inputs=sample();inputs['link_valid'].zero_()
    out=net(**inputs)
    assert not out['reliable'].any()
    assert torch.isfinite(out['match_logit']).all()


def test_candidate_trust_uses_verified_box_labels_without_claiming_view_absence():
    net=LandmarkBeliefNetwork(text_dim=12,vision_dim=16,hidden=16,heads=2)
    out=net(**sample());out['candidate_match_logits'].retain_grad()
    labels=dict(ranking=torch.tensor([[1.,0,0,0,0]]*2),rank_label_mask=torch.tensor([[True,True,False,False,False]]*2),
                angle=torch.zeros(2,3,2),angle_valid=torch.zeros(2,3,dtype=torch.bool),distance=torch.zeros(2,3),
                offset=torch.zeros(2,5,2),offset_valid=torch.tensor([[True,False,False,False,False]]*2),
                match=torch.ones(2),match_valid=torch.zeros(2,dtype=torch.bool))
    loss,_=supervision_loss(out,labels,refine=False);loss.backward()
    grad=out['candidate_match_logits'].grad
    assert grad[:,0].abs().sum()>0 and grad[:,1].abs().sum()>0
    assert (grad[:,2:]==0).all()
