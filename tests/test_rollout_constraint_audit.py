from graphnav.closedloop import ClosedLoopProtocol
from scripts.audit_rollout_constraints import pose_constraints


def test_source_invalid_altitude_requires_bounded_nonworsening_recovery():
    protocol=ClosedLoopProtocol()
    ok=pose_constraints([[0,0,125,0],[10,0,115,0],[20,0,105,0],[30,0,95,0]],0,(-100,-100,100,100),protocol)
    assert ok['initial_altitude_outside'] and not ok['altitude_outside_worsening'] and not ok['vertical_step_violations'] and not ok['new_illegal_altitude']
    bad=pose_constraints([[0,0,125,0],[60,0,140,0]],0,(-100,-100,100,100),protocol)
    assert bad['altitude_outside_worsening']==1 and bad['horizontal_step_violations']==1 and bad['vertical_step_violations']==1


def test_legal_start_is_not_exempted_from_new_outside_state():
    result=pose_constraints([[0,0,95,0],[110,0,105,0]],0,(-100,-100,100,100),ClosedLoopProtocol())
    assert result['new_illegal_xy']==1 and result['new_illegal_altitude']==1
