from types import SimpleNamespace
import numpy as np
import pytest
from graphnav.closedloop import (PublicEpisode, ClosedLoopProtocol, PublicCatalog,
                                score_trace, aggregate, rollout)


def test_metrics_distinguish_oracle_from_final_success():
    row=score_trace([[0,0,30],[90,0,30],[0,0,30]], [100,0,0],
                    [[0,0],[100,0]], 'policy_stop')
    assert row['OSR']==1 and row['SR']==0 and row['SPL']==0
    assert row['false_stop']==1 and row['NE']==100
    assert aggregate([row])['False_Stop_Rate']==1


def test_benchmark_xy_boundary_and_spl():
    row=score_trace([[0,0,100],[80,0,100]], [100,0,0], [[0,0],[80,0]],'horizon')
    assert row['SR']==1 and row['SPL']==1 and row['nDTW']==1 and row['sDTW']==1
    assert aggregate([row])['False_Stop_Rate'] is None
    row=score_trace([[0,0],[0,100],[100,0]], [100,0], [[0,0],[100,0]],'policy_stop')
    assert row['SPL']==pytest.approx(100/(100+np.sqrt(20000)))


def test_public_rollout_stop_has_no_movement_or_label_access():
    public=PublicEpisode('map','go east',np.array([0.,0.,30.,0.]),())
    class Builder:
        def transform(self,name): return SimpleNamespace(x_min=0,y_min=0,x_max=200,y_max=200)
        def build_from_poses(self,record,poses):
            assert not hasattr(record,'target_position') and not hasattr(record,'trajectory')
            assert not hasattr(record,'object_id')
            assert len(poses)==1
            return SimpleNamespace(pose=poses[-1])
    class Controller:
        def step(self,*args,**kwargs): raise AssertionError('must not execute after stop')
    def policy(state,text):
        assert text=='go east'
        return SimpleNamespace(xyz=np.array([10,0,30]),diagnostics=None,should_stop=True)
    poses,termination,timings,_=rollout(public,Builder(),Controller(),policy,0,ClosedLoopProtocol())
    assert len(poses)==1 and termination=='policy_stop' and len(timings)==1


def test_catalog_never_resolves_target_id_during_rollout():
    class Catalog:
        def referenced_landmarks(self,record): raise AssertionError('label access')
    public=PublicEpisode('m','query',np.zeros(4),('bound polygon',))
    assert PublicCatalog(Catalog()).referenced_landmarks(public)==public.references


def test_metrics_reject_nonfinite_inputs():
    with pytest.raises(ValueError): score_trace([[0,np.nan]],[0,0],[[0,0]],'horizon')


def test_a3_public_map_binding_keeps_homonymous_road_fragments_without_target():
    from graphnav.a3_runtime import public_static_record
    objects={1:dict(name='Main Road',object_type='TrafficRoad',contour=[[0,0],[10,0],[10,10],[0,10]]),
             2:dict(name='Main Road',object_type='TrafficRoad',contour=[[20,0],[30,0],[30,10],[20,10]])}
    episode=PublicEpisode('map','near Main Road',np.zeros(4),(),('Main Road',))
    transform=SimpleNamespace(x_min=0,y_min=0,x_max=100,y_max=100)
    record=public_static_record(episode,transform,SimpleNamespace(objects={'map':objects}))
    assert len(record['landmarks'])==2 and record['target_xy']==[0.,0.]
    assert all('id' not in landmark for landmark in record['landmarks'])
