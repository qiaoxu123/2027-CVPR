import pytest
from scripts.summarize_closedloop import KEYS, paired_stats


def row(index, success):
    return dict(index=index, episode_id=['map', index], map_name='map', poses=[[0, 0, 50, 0]],
        metrics={k: success for k in KEYS})


def test_paired_bootstrap_preserves_gains_and_rejects_changed_starts():
    a = {0: row(0, 0), 1: row(1, 1)}
    b = {0: row(0, 1), 1: row(1, 1)}
    result = paired_stats(a, b, draws=20)
    assert result['gained'] == [0] and result['lost'] == []
    assert result['episode_weighted']['SR']['delta'] == .5
    assert result['equal_map_weighted']['SR']['ci95'] == [.5, .5]
    b[0]['poses'][0][0] = 1
    with pytest.raises(ValueError, match='start differs'):
        paired_stats(a, b, draws=20)
