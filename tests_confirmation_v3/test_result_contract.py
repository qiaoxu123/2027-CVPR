import copy
import pytest
from experiments.a2_waldo_landmark_belief.relation_v3_calibrate import validate_result_contract
from experiments.a2_waldo_landmark_belief.calibration_protocol import verify_gate


def record(selected=None,stats=None):
    return dict(status='v3_candidate_requires_real_closed_loop_pilot' if selected else 'v3_FIT_no_candidate_VERIFY_not_consumed',
        enabled=False,gate_final=False,fixed_action=.25,temperature=1.,temperature_fitting=False,benchmark_used=False,no_verify_reselection=True,
        fit_groups=84,verify_groups=209,fit_selected_parameters=selected,verification=stats,verification_performance_evaluated=stats is not None,
        verify_gate=verify_gate(stats) if stats else dict(passed=False,checks=dict(fit_candidate_exists=False)),minimum_gain=1.,maximum_harm=0.)


def test_retained_no_fit_result_cannot_claim_enabled_or_hidden_verify():
    r=record();decision=dict(selected_parameters=None);validate_result_contract(r,decision)
    for key,value in (('enabled',True),('minimum_gain',.1),('verification_performance_evaluated',True)):
        bad=copy.deepcopy(r);bad[key]=value
        with pytest.raises(ValueError):validate_result_contract(bad,decision)


def test_failed_verify_cannot_be_relabelled_as_enabled_or_wrong_population():
    selected=dict(minimum_gain=.9,maximum_harm=.1,temperature=1.)
    stats=dict(groups=209,supported_groups=30,a0_correct_groups=100,gain_groups=5,gain_ci95=[0.,.1],broken_group_wilson_upper95=.02,mean_ne_difference_m=-.1)
    r=record(selected,stats);decision=dict(selected_parameters=selected);validate_result_contract(r,decision)
    bad=copy.deepcopy(r);bad.update(enabled=True,minimum_gain=.9,maximum_harm=.1)
    with pytest.raises(ValueError):validate_result_contract(bad,decision)
    bad=copy.deepcopy(r);bad['verification']['groups']=208
    with pytest.raises(ValueError):validate_result_contract(bad,decision)
