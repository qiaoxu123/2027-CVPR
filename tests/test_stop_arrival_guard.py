from types import SimpleNamespace
import pytest
from graphnav.trajectory_policy import stop_arrival_allowed
from scripts.evaluate_closedloop import configure_stage1_controls


def test_original_millimeter_gate_and_meter_scale_interventions():
    assert stop_arrival_allowed(.001)
    assert not stop_arrival_allowed(.002)
    assert not stop_arrival_allowed(4.9,1)
    assert stop_arrival_allowed(4.9,5)
    assert stop_arrival_allowed(20,20)
    assert not stop_arrival_allowed(20.001,20)
    assert stop_arrival_allowed(1000,-1)
    assert stop_arrival_allowed(None)


@pytest.mark.parametrize('changes',[
    dict(split='val_unseen'),dict(split='test_unseen'),dict(stage1_diagnostic=False),
    dict(stage1_arrival_guard_m=-2),dict(stage1_arrival_guard_m=float('nan')),dict(stage1_grid_size=384)])
def test_no_heldout_tuning_and_invalid_guard_values(changes):
    args=dict(stage1_diagnostic=True,split='val_seen',stop_threshold=.3,stage1_grid_size=None,stage1_arrival_guard_m=5)
    args.update(changes)
    with pytest.raises(ValueError):configure_stage1_controls(SimpleNamespace(**args))


def test_guard_radius_does_not_replace_probability_threshold():
    args=SimpleNamespace(stage1_diagnostic=True,split='val_seen',stop_threshold=.3,stage1_grid_size=None,stage1_arrival_guard_m=5)
    configure_stage1_controls(args)
    assert args.stop_threshold==.3 and args.stage1_arrival_guard_m==5
