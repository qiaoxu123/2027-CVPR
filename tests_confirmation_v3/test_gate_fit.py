import pytest
import torch
from experiments.a2_waldo_landmark_belief.calibration_protocol import calibration_fold,verify_gate
from experiments.a2_waldo_landmark_belief.relation_v3_gate_fit import choose_fit,evaluate_verify


def packets(fold,count,gains):
    keys=[('m',i) for i in range(2000) if calibration_fold('m',i)==fold][:count]
    return [dict(group=list(k),fold=fold,episode=n,step=0,available=True,changed=True,
        before_ne=30. if n<gains else 10.,action_ne=10.,scores=torch.tensor([.09,.9,.01]) if n<gains else torch.tensor([.85,.05,.1])) for n,k in enumerate(keys)]


def test_fit_uses_complete84_groups_and_rejects_verify_or_missing_targets():
    fit=packets('fit',84,20);expected={tuple(p['group']) for p in fit}
    choices,chosen=choose_fit(fit,expected)
    assert len(choices)==30 and chosen is not None and chosen['parameters']['temperature']==1.
    with pytest.raises(ValueError):choose_fit(fit[:-1],expected)
    bad=[dict(p) for p in fit];bad[0]['fold']='verify'
    with pytest.raises(ValueError):choose_fit(bad,expected)


def test_no_fit_candidate_never_iterates_heldout_packets():
    class Forbidden:
        def __iter__(self):raise AssertionError('Heldout performance consumed before FIT choice')
    with pytest.raises(ValueError):evaluate_verify(Forbidden(),None,set())
    fit=packets('fit',84,0);_,chosen=choose_fit(fit,{tuple(p['group']) for p in fit})
    assert chosen is None


def test_single_frozen_choice_scores_all209_groups_and_preserves_six_checks():
    verify=packets('verify',209,10);expected={tuple(p['group']) for p in verify}
    result=evaluate_verify(verify,dict(minimum_gain=0.,maximum_harm=1.,temperature=1.),expected)
    assert result['groups']==209 and verify_gate(result)['passed']
    with pytest.raises(ValueError):evaluate_verify(verify[:-1],dict(minimum_gain=0.,maximum_harm=1.,temperature=1.),expected)


def test_invalid_verify_scores_cannot_turn_into_silent_fallback():
    verify=packets('verify',209,10);expected={tuple(p['group']) for p in verify}
    verify[100]['scores'][1]=float('nan')
    with pytest.raises(ValueError):evaluate_verify(verify,dict(minimum_gain=0.,maximum_harm=1.,temperature=1.),expected)
