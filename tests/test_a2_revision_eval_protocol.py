import numpy as np,pytest
from experiments.a2_waldo_landmark_belief.revision_eval_protocol import sensor_payload,target_observation_coverage,recall_row,pilot_decision


def test_sensor_visibility_and_candidate_coverage_are_posthoc_and_reproducible():
    observation=dict(bgr=np.ones((384,384,3),dtype=np.uint8),footprint_world=np.array([[0.,0.],[40.,0.],[40.,40.],[0.,40.]]))
    sensor=sensor_payload(observation,[dict(kind=0,poly=[[18,18],[22,18],[22,22],[18,22]])])
    c=target_observation_coverage(sensor,np.array([20.,20.]),0)
    assert c['target_visible'] and c['candidate_point_covered']
    c=target_observation_coverage(sensor,np.array([20.,20.]),1)
    assert c['target_visible'] and not c['candidate_point_covered']
    c=target_observation_coverage(sensor,np.array([100.,100.]),0)
    assert not c['target_visible']


def test_recall_uses_original_first_and_last_observation_and_all_k():
    points=[[float(i),0.] for i in range(16)]
    diagnostics=[dict(static_top16_world_xy=points,belief_top16_world_xy=points),dict(static_top16_world_xy=points,belief_top16_world_xy=list(reversed(points)))]
    r=recall_row(diagnostics,np.array([35.,0.]))
    assert r['dynamic_final_R@1/20m']==1 and r['dynamic_first_R@1/20m']==0


def test_pilot_needs_real_intervention_and_no_sr_ne_regression():
    a=[dict(metrics=dict(SR=0.,NE=40.),poses=[[0,0,50],[1,0,50]],step_diagnostics=[])]
    b=[dict(metrics=dict(SR=1.,NE=10.),poses=[[0,0,50],[30,0,50]],step_diagnostics=[dict(a2_updated=True)])]
    assert pilot_decision(a,b)['passed']
    assert not pilot_decision(a,a)['passed']
    assert not pilot_decision(b,a)['passed']


def test_formal_evaluation_rejects_partial_or_duplicate_episode_population():
    from experiments.a2_waldo_landmark_belief.revision_eval_protocol import require_population
    with pytest.raises(ValueError):require_population('val_seen',[dict(index=i) for i in range(2469)])
    entries=[dict(index=i) for i in range(2470)];require_population('val_seen',entries)
    entries[-1]['index']=0
    with pytest.raises(ValueError):require_population('val_seen',entries)


def test_committed_trace_detects_mutation_and_uncommitted_trace_is_not_trusted(tmp_path):
    import json
    from experiments.a2_waldo_landmark_belief.revision_eval_protocol import write_trace,read_trace
    p=tmp_path/'episode.json';row=dict(index=1,signature_sha256='sig',metrics=dict(SR=1.))
    write_trace(row,p);assert read_trace(p,'sig',1)==row
    p.write_text(json.dumps(dict(row,metrics=dict(SR=0.))))
    with pytest.raises(ValueError):read_trace(p,'sig',1)


def test_saved_closed_loop_poses_must_follow_original_controller_commands():
    from experiments.a2_waldo_landmark_belief.revision_eval_protocol import controller_consistency
    from graphnav.closedloop import ClosedLoopProtocol
    row=dict(poses=[[0.,0.,50.,0.],[50.,0.,50.,0.]],termination='horizon',
             step_diagnostics=[dict(commanded_xyz=[100.,0.,50.],commanded_stop=False)])
    assert controller_consistency(row,(0.,0.,100.,100.),0.,ClosedLoopProtocol(horizon=1))
    row['poses'][1][0]=100.
    with pytest.raises(ValueError):controller_consistency(row,(0.,0.,100.,100.),0.,ClosedLoopProtocol(horizon=1))
