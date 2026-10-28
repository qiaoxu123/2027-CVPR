import torch
from experiments.a2_waldo_landmark_belief.direct_risk_head import DirectRiskHead
from experiments.a2_waldo_landmark_belief.direct_risk_runtime import risk_forward


def test_runtime_uses_shared_public_features_and_exact_OFF_return():
    spatial=dict(base_logits=torch.tensor([.1,0.]),grid_world=torch.tensor([[0.,0.],[100.,0.]]),
        candidate_xy=torch.tensor([[100.,0.]]),candidate_support=torch.tensor([[False,True]]))
    rank=dict(scores=torch.tensor([[0.]]),candidate_match_logits=torch.tensor([[10.]]),reliable=torch.tensor([[True]]),benefit_logit=torch.tensor([0.]))
    head=DirectRiskHead(torch.zeros(15),torch.ones(15),torch.ones(3))
    with torch.no_grad():head.linear.weight.zero_();head.linear.bias.copy_(torch.tensor([-10.,10.,-10.]))
    off,info,tensors=risk_forward(spatial,rank,head,dict(enabled=False))
    assert off is spatial['base_logits'] and not info['updated'] and tensors['features'].shape==(1,15)
    mixed,info,tensors=risk_forward(spatial,rank,head,dict(enabled=True,minimum_gain=.5,maximum_harm=.1,temperature=1.))
    assert mixed.argmax()==1 and info['gate']==.25
    assert torch.equal(mixed,tensors['fixed_action_logits'])
    assert set(tensors)=={'q','available','features','head_logits','corrected_logits','scores','fixed_action_logits'}
