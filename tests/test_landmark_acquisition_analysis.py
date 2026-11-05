"""Diagnostic scoring never converts OSR into actual SR."""
from scripts.analyze_landmark_acquisition import summarize_episode, paired_changes


def row(i=0, method='Geometry', success=0, osr=1, goal=25):
    return dict(index=i, method=method, termination='horizon',
        metrics=dict(SR=success, OSR=osr, initial_polygon_distance_m=100., path_length_m=200., first_entry_step=2),
        step_diagnostics=[dict(selected_goal_xy=[0, 0], public_landmark_goal_distance_m=goal),
                          dict(selected_goal_xy=[40, 0], public_landmark_goal_distance_m=goal)],
        decision_seconds=[.1, .2])


def test_osr_is_not_stop_success_and_no_goal_does_not_imply_pure_stop_bug():
    r = summarize_episode(row())
    assert r['SPL_polygon'] == 0
    assert r['entered_then_left'] == 1
    assert r['failure_observation'] == 'entered_then_left_wrong_goal'
    assert r['correct_stop'] == 0
    assert r['goal_jumps_over20'] == 1


def test_paired_counts_do_not_hide_damage():
    rows = [row(0, success=0), row(1, success=1),
            row(0, 'TrackedPGVB', success=1), row(1, 'TrackedPGVB', success=0)]
    change = paired_changes(rows, 'Geometry', 'TrackedPGVB')
    assert change['corrected'] == [0]
    assert change['damaged'] == [1]
    assert change['net'] == 0
