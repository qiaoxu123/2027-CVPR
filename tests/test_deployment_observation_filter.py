import pytest
from graphnav.deployment_observation_filter import observation_in_envelope


def frame(height,step=1):
    return {'agent_pose':[0.,0.,height,1.2],'step':step}


def test_teacher_frames_obey_action_envelope_but_original_initial_pose_is_retained():
    assert not observation_in_envelope(frame(2.),0.)
    assert not observation_in_envelope(frame(150.),0.)
    assert observation_in_envelope(frame(10.),0.)
    assert observation_in_envelope(frame(100.),0.)
    assert observation_in_envelope(frame(150.,0),0.)
    assert observation_in_envelope(frame(2.,0),0.)
    assert not observation_in_envelope(frame(0.,0),0.)


def test_filter_does_not_need_target_labels_or_change_input_pose():
    f=frame(25.);copy=dict(f)
    assert observation_in_envelope(f,0.) and f==copy
    with pytest.raises(ValueError):observation_in_envelope(frame(float('nan')),0.)


def test_aligned_training_pose_preserves_xy_yaw_and_initial_observation():
    from graphnav.deployment_observation_filter import aligned_observation_pose
    low={'agent_pose':[12.,34.,105.,1.2],'step':10}
    assert aligned_observation_pose(low,100.)==[12.,34.,110.,1.2]
    assert low['agent_pose'][2]==105.
    high={'agent_pose':[12.,34.,250.,-2.1],'step':10}
    assert aligned_observation_pose(high,100.)==[12.,34.,200.,-2.1]
    high['step']=0
    assert aligned_observation_pose(high,100.)==high['agent_pose']
