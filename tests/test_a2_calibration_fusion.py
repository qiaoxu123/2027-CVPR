import torch
from experiments.a2_waldo_landmark_belief.calibration_fusion import calibrated_logits


def packet():
    return dict(output=dict(scores=torch.tensor([2.,0.]),reliable=torch.tensor([True,True]),
                            candidate_match_logits=torch.tensor([2.,-2.]),match_logit=torch.tensor(2.),offset=torch.zeros(2,2)),
                spatial=dict(base_logits=torch.tensor([0.,1.,-1.]),grid_world=torch.tensor([[0.,0.],[4.,0.],[20.,0.]]),
                             candidate_xy=torch.tensor([[0.,0.],[4.,0.]]),candidate_support=torch.tensor([[True,True,False],[False,True,False]])))


def test_disabled_or_low_trust_is_exact_a0_and_outside_support_is_unchanged():
    p=packet();params=dict(enabled=False,alpha=.5,threshold=.5,temperature=1.)
    assert calibrated_logits(p,params) is p['spatial']['base_logits']
    params['enabled']=True;params['threshold']=.99
    assert calibrated_logits(p,params) is p['spatial']['base_logits']
    params['threshold']=.5;result=calibrated_logits(p,params)
    assert result[-1]==p['spatial']['base_logits'][-1]
    assert torch.allclose(result.exp().sum(),p['spatial']['base_logits'].exp().sum(),atol=1e-6)


def test_calibration_analysis_scores_fixed_choices_without_gpu_or_private_input_to_fusion():
    from experiments.a2_waldo_landmark_belief.calibrate import evaluate_choices
    p=dict(group=['m',0],target_xy=torch.zeros(2),
           output=dict(scores=torch.tensor([5.,0.]),reliable=torch.tensor([True,True]),
                       candidate_match_logits=torch.tensor([5.,5.]),match_logit=torch.tensor(5.),offset=torch.zeros(2,2)),
           spatial=dict(base_logits=torch.tensor([0.,.1]),grid_world=torch.tensor([[0.,0.],[100.,0.]]),
                        candidate_xy=torch.tensor([[0.,0.],[100.,0.]]),candidate_support=torch.eye(2,dtype=torch.bool)))
    summaries=evaluate_choices([p],[dict(alpha=0.,threshold=.5),dict(alpha=.5,threshold=.5)],1.,device='cpu')
    assert summaries[0]['mean_r1_difference']==0 and summaries[1]['mean_r1_difference']==1
