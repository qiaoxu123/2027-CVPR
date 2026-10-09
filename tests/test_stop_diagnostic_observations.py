from graphnav.closedloop import ClosedLoopProtocol
from scripts.audit_stop_failures import classify


def trace(poses, decisions, success, oracle, diagnostics=None):
    return dict(poses=poses, decision_seconds=[0.] * decisions,
                step_diagnostics=diagnostics or [{} for _ in range(decisions)],
                metrics=dict(SR=success, OSR=oracle), termination='horizon')


def test_executed_horizon_endpoint_is_not_observed():
    row = trace([[100, 0, 20, 0], [50, 0, 20, 0], [0, 0, 20, 0]], 2, 1, 1)
    result = classify(row, [0, 0], 0, ClosedLoopProtocol(), .3)
    assert result['success']
    assert result['observed_target_near_frames'] == 0


def test_invalid_height_visit_is_not_a_legal_stop_opportunity():
    row = trace([[0, 0, 120, 0], [50, 0, 110, 0]], 2, 0, 1)
    result = classify(row, [0, 0], 0, ClosedLoopProtocol(), .3)
    assert result['failed_after_only_illegal_observed_arrival']
    assert not result['failed_after_observed_legal_arrival']


def test_predicted_goal_guard_and_stop_probability_are_separate():
    row = trace([[0, 0, 20, 0], [50, 0, 20, 0]], 2, 0, 1,
                [dict(stop_probability=.8, predicted_goal_distance=50), {}])
    result = classify(row, [0, 0], 0, ClosedLoopProtocol(), .3)
    assert result['failed_after_observed_legal_arrival']
    assert result['predicted_goal_gate_rejection_frames'] == 1
    assert result['probability_rejection_frames'] == 0


def test_native_sbf_trace_without_head_diagnostics_is_explicit():
    row = trace([[0, 0, 20, 0], [50, 0, 20, 0]], 2, 0, 1)
    row.pop('step_diagnostics')
    result = classify(row, [0, 0], 0, ClosedLoopProtocol(), .3)
    assert result['failed_after_observed_legal_arrival']
    assert not result['decision_diagnostics_available']
    assert result['legal_arrival_head_frames'] == 0
    assert result['max_stop_probability_at_legal_arrival'] is None
