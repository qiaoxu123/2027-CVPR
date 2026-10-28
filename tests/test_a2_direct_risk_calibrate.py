import pytest
import torch
from experiments.a2_waldo_landmark_belief.direct_risk_calibrate import fit_choice,evaluate_risk
from experiments.a2_waldo_landmark_belief.calibration_protocol import calibration_fold


def packets():
    ids=[i for i in range(100) if calibration_fold('m',i)=='fit'][:24]
    return [dict(group=['m',oid],fold='fit',scores=torch.tensor([.05,.9,.05] if i<12 else [.05,.05,.9]),
        available=True,changed=True,before_ne=40. if i<12 else 0.,action_ne=0. if i<12 else 40.) for i,oid in enumerate(ids)]


def test_fit_selects_one_safe_point_and_never_consumes_verify_targets():
    rows=packets();choices,chosen=fit_choice(rows)
    assert len(choices)==30 and chosen is not None
    stats=evaluate_risk(rows,chosen['parameters'])
    assert stats['gain_groups']==12 and stats['broken_groups']==0
    rows[0]=dict(rows[0],fold='verify')
    with pytest.raises(ValueError):fit_choice(rows)


def test_empty_evidence_stays_OFF_without_fabricated_candidate():
    rows=[dict(p,available=False,changed=False) for p in packets()]
    assert fit_choice(rows)==([],None)


def test_private_target_changes_outcome_only_not_public_gate_scores():
    from experiments.a2_waldo_landmark_belief.direct_risk_head import DirectRiskHead
    from experiments.a2_waldo_landmark_belief.direct_risk_runtime import risk_forward
    from experiments.a2_waldo_landmark_belief.direct_risk_calibrate import make_packet
    spatial=dict(base_logits=torch.tensor([.1,0.]),grid_world=torch.tensor([[0.,0.],[100.,0.]]),
        candidate_xy=torch.tensor([[100.,0.]]),candidate_support=torch.tensor([[False,True]]))
    rank=dict(scores=torch.tensor([[0.]]),candidate_match_logits=torch.tensor([[10.]]),reliable=torch.tensor([[True]]),benefit_logit=torch.tensor([0.]))
    head=DirectRiskHead(torch.zeros(15),torch.ones(15),torch.ones(3))
    _,_,risk=risk_forward(spatial,rank,head,dict(enabled=False))
    row=dict(public=dict(spatial=spatial),prediction={k:v[0] for k,v in rank.items()},risk=risk,
        labels=dict(target_xy=torch.tensor([100.,0.])),target_instance=['m',1],fold='fit',episode=1,step=0)
    gain=make_packet(row,head);row['labels']['target_xy']=torch.tensor([0.,0.]);harm=make_packet(row,head)
    assert gain['before_ne']==100 and gain['action_ne']==0 and harm['action_ne']==100
    assert torch.equal(gain['scores'],harm['scores'])
    row['risk']=dict(risk,fixed_action_logits=spatial['base_logits'])
    with pytest.raises(ValueError):make_packet(row,head)
