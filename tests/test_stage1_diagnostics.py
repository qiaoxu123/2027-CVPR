from types import SimpleNamespace
import pytest
from scripts.evaluate_closedloop import configure_stage1_controls


def controls(**changes):
    values=dict(stage1_diagnostic=True,split='val_seen',stop_threshold=.7,stage1_grid_size=384)
    values.update(changes)
    return SimpleNamespace(**values)


def test_diagnostic_stop_threshold_is_preserved_and_formal_default_is_frozen():
    args=controls()
    configure_stage1_controls(args)
    assert args.stop_threshold==.7
    assert args.stage1_grid_size==384
    assert args.head_execution=='guarded-blend'
    args=controls(stage1_diagnostic=False,stage1_grid_size=None)
    configure_stage1_controls(args)
    assert args.stop_threshold==.3


@pytest.mark.parametrize('changes',[
    dict(split='test_unseen'),dict(split='val_unseen'),dict(stop_threshold=float('nan')),
    dict(stop_threshold=0),dict(stop_threshold=1),dict(stage1_grid_size=1),
    dict(stage1_grid_size=385),dict(stage1_diagnostic=False),
])
def test_invalid_or_heldout_interventions_are_rejected(changes):
    with pytest.raises(ValueError):
        configure_stage1_controls(controls(**changes))
