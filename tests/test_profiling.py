import json
from types import SimpleNamespace
import pytest
from graphnav.closedloop import PublicEpisode, ClosedLoopProtocol, rollout
from graphnav.profiling import DecisionProfiler
import numpy as np


def test_nested_stages_are_exclusive_and_first_sample_is_reported(tmp_path):
    profiler=DecisionProfiler(lambda:None)
    for step in range(2):
        with profiler.sample('map',step):
            with profiler.phase('visual_residual_history_ms'):
                with profiler.phase('trajectory_stop_ms'):
                    sum(range(1000))
    profiler.save(tmp_path)
    report=json.loads((tmp_path/'profile.json').read_text())
    assert report['samples']==2
    assert report['first']['step']==0
    assert report['excluding_first']['rollout_ms']['population_variance']==0
    for row in profiler.rows:
        assert row['model_components_ms']+row['environment_ms']+row['other_rollout_ms']==pytest.approx(row['rollout_ms'])
        assert row['visual_residual_history_ms']>=0


def test_stop_break_still_records_observation_and_decision():
    profiler=DecisionProfiler(lambda:None)
    builder=SimpleNamespace(transform=lambda name:SimpleNamespace(x_min=0,y_min=0,x_max=1,y_max=1),
        build_from_poses=lambda *args:None)
    profiler.wrap(builder,'build_from_poses','environment_ms')
    episode=PublicEpisode('map','instruction',np.array([0.,0.,10.]),())
    result=rollout(episode,builder,None,
        lambda *args:SimpleNamespace(should_stop=True,diagnostics={}),0,
        ClosedLoopProtocol(),profiler=profiler)
    assert result[1]=='policy_stop'
    assert len(profiler.rows)==1
    assert profiler.rows[0]['environment_ms']>0
