import pytest
import torch
from experiments.rgb_grounding.head import VisualEvidenceNetwork,Evidence,grounding_loss,update_belief
from experiments.rgb_grounding.city_features import visual_phrase


def sample():
    return dict(text=torch.randn(2,16),images=torch.randn(2,3,16),patch_tokens=torch.randn(2,3,4,16),language_tokens=torch.randn(2,5,16),language_valid=torch.ones(2,5,dtype=torch.bool),quality=torch.rand(2,3,3),valid=torch.tensor([[True,True,False],[False,False,False]]))


@pytest.mark.parametrize('tokens',[False,True])
def test_private_rejection_and_unknown(tokens):
    model=VisualEvidenceNetwork(16,8,tokens);public=sample();e=model(public)
    assert torch.isfinite(e.score).all() and not e.confidence[~public['valid']].any()
    with pytest.raises(ValueError):model(dict(public,target_object_id=1))


def test_unknown_negative_has_no_gradient():
    score=torch.tensor([[2.,1.,40.]],requires_grad=True)
    e=Evidence(score,torch.ones_like(score)*.5,torch.ones_like(score,dtype=torch.bool))
    loss=grounding_loss(e,torch.tensor([[True,False,False]]),torch.tensor([[True,True,False]]),torch.ones(1,3,3))['total'];loss.backward()
    assert score.grad[0,2]==0 and score.grad[0,0]<0


def test_multipositive_and_absent_negative():
    e=Evidence(torch.randn(1,3,requires_grad=True),torch.ones(1,3)*.5,torch.ones(1,3,dtype=torch.bool))
    assert grounding_loss(e,torch.tensor([[True,True,False]]),torch.tensor([[True,True,False]]),torch.ones(1,3,3))['usable']==0


def test_unknown_static_logits_unchanged():
    static=torch.randn(1,6);e=Evidence(torch.tensor([[1.,3.]]),torch.ones(1,2),torch.ones(1,2,dtype=torch.bool))
    association=torch.zeros(1,2,6,dtype=torch.bool);association[0,0,1]=True;association[0,1,3]=True
    updated,known=update_belief(static,e,association)
    assert torch.equal(updated[~known],static[~known])
    assert not torch.equal(updated[known],static[known])


def test_nominal_attribute_binding():
    phrase,supported=visual_phrase('A gray building with a white car beside it.')
    assert supported and 'gray' in phrase and 'white' not in phrase


def test_unresolved_language_not_fabricated():
    language='Return to the launch location.'
    assert visual_phrase(language)==(language,False)


def test_rgb_intervention_keeps_destination_polygon():
    import numpy as np
    from experiments.rgb_grounding.pixel_controls import intervention
    tight=np.full((12,18,3),64,np.uint8);mask=np.zeros((12,18),bool);mask[2:9,4:15]=True
    donor=np.full((30,8,3),200,np.uint8)
    shuffled=intervention(tight,mask,donor);zero=intervention(tight,mask)
    assert shuffled.shape==tight.shape and np.all(shuffled[mask]==200)
    assert np.all(zero[mask]==0)
    assert np.all(shuffled[~mask]==127) and np.array_equal(shuffled[~mask],zero[~mask])


def test_unknown_gallery_has_no_semantic_metric():
    from experiments.rgb_grounding.evaluate_features import metrics
    row={'labels':{'positive_roi_mask':torch.zeros(2,dtype=torch.bool),'semantic_valid_mask':torch.zeros(2,dtype=torch.bool)},'public':{'roi_valid':torch.ones(2,dtype=torch.bool)},'metadata':{'roi_objects':[{'object_id':'a'},{'object_id':'b'}]}}
    assert metrics(torch.tensor([1.,2.]),row)['r1'] is None


def test_future_feature_cannot_be_relabelled_as_current():
    from experiments.rgb_grounding.goal_evidence import predict
    current={'sample_id':'current','episode_id':'ep','map_name':'map','cutoff_step':5,'actual_camera_pose':[1.,2.,3.,0.],'roi_objects':[]}
    future=dict(current,cutoff_step=8)
    dataset=[{'metadata':current,'public':{},'labels':{}}]
    record={'row':{'metadata':future,'public':{'roi_quality':torch.empty(0,3)}}}
    with pytest.raises(ValueError,match='current observation differs'):
        predict(dataset,{'current':record},1,0,True)
