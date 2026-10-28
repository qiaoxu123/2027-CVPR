import json
import pytest
from experiments.a2_waldo_landmark_belief.relation_v3_confirmation_pipeline import after_fit,predecessor_ready


def test_no_fit_choice_never_launches_replay_or_verify(tmp_path):
    (tmp_path/'calibration').mkdir();(tmp_path/'calibration/fit_decision.json').write_text(json.dumps(dict(selected_parameters=None)))
    def forbidden(*a,**k):raise AssertionError('Heldout phase launched despite no FIT choice')
    state=after_fit(tmp_path,forbidden,lambda:dict(enabled=False,verification=None,verification_performance_evaluated=False))
    assert state=='FIT_no_candidate_VERIFY_unconsumed'


def test_frozen_fit_runs_replay_before_the_only_verify(tmp_path):
    (tmp_path/'calibration').mkdir()
    (tmp_path/'calibration/fit_decision.json').write_text(json.dumps(dict(selected_parameters=dict(minimum_gain=.8,maximum_harm=.1,temperature=1.))))
    called=[]
    state=after_fit(tmp_path,lambda name:called.append(name),lambda:dict(enabled=False,verification=dict(groups=209),verification_performance_evaluated=True))
    assert called==['replay','verify'] and state=='VERIFY_rejected_no_forced_enable'


def test_completed_candidate_is_retained_without_waiting_for_GPU(tmp_path):
    (tmp_path/'calibration').mkdir()
    (tmp_path/'calibration/fit_decision.json').write_text(json.dumps(dict(selected_parameters=dict(minimum_gain=.8,maximum_harm=.1,temperature=1.))))
    (tmp_path/'calibration/candidate_calibration.json').write_text('{}')
    def forbidden(*a,**k):raise AssertionError('Completed final decision needlessly reacquired GPU')
    result=after_fit(tmp_path,forbidden,lambda:dict(enabled=True,verification=dict(groups=209),verification_performance_evaluated=True))
    assert result=='calibration_passed_real_pilot_pending'


def test_predecessor_requires_successful_terminal_and_complete_files():
    service=dict(LoadState='loaded',ActiveState='active',SubState='running',MainPID='17',ExecMainStatus='0',Result='success')
    assert not predecessor_ready(service,True)
    service.update(MainPID='0',SubState='exited')
    assert predecessor_ready(service,True)
    with pytest.raises(ValueError):predecessor_ready(service,False)
    service.update(ActiveState='failed',SubState='failed',ExecMainStatus='1',Result='exit-code')
    with pytest.raises(ValueError):predecessor_ready(service,True)
