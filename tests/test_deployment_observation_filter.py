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
