from types import ModuleType,SimpleNamespace
import sys
import numpy as np
import torch
from graphnav.a3_runtime import A3TrajectoryPolicy
from graphnav.full_localization import GoalField
from graphnav.trajectory_policy import TrajectoryOutput


def test_execution_ablations_keep_public_goal_and_candidate_stop_consistent(monkeypatch):
    package=ModuleType('sbfnav');maps=ModuleType('sbfnav.mapdata');maps.GROUND_LEVEL={'map':0.}
    monkeypatch.setitem(sys.modules,'sbfnav',package);monkeypatch.setitem(sys.modules,'sbfnav.mapdata',maps)
    policy=A3TrajectoryPolicy.__new__(A3TrajectoryPolicy);policy.device=torch.device('cpu')
    policy.visited_goal_cells=set();policy.explore_candidates=0
    policy.record=dict(world_bounds=[0,0,400,400],map_name='map');policy.stop_threshold=.3;policy.stop_positive_weight=20.
    field=GoalField(torch.tensor([[2.,1.]]),torch.tensor([[0,1]]),torch.tensor([[[.2,.3],[.8,.8]]]),torch.zeros(1,2,16))
    policy.frozen_field=lambda:(field,{})
    paths=torch.zeros(1,2,3,8,3);paths[0,0,1,0]=torch.tensor([100.,120.,30.]);paths[0,1,0,0]=torch.tensor([300.,320.,30.])
    output=TrajectoryOutput(paths,torch.tensor([[[0.,2.,0.],[2.,0.,0.]]]),torch.tensor([[-1.,4.]]),torch.tensor([[4.,6.]]))
    policy.head=lambda **kwargs:output
    state=SimpleNamespace(pose=np.array([10,10,30,0]))
    policy.execution='joint';assert policy.predict(state,'near hall').diagnostics['candidate']==1
    policy.execution='fixed-goal';prediction=policy.predict(state,'near hall')
    np.testing.assert_allclose(prediction.xyz,[100,120,30]);assert prediction.should_stop
    policy.execution='fixed-goal-no-stop';assert not policy.predict(state,'near hall').should_stop
    policy.execution='direct-goal';prediction=policy.predict(state,'near hall')
    np.testing.assert_allclose(prediction.xyz,[80,120,30]);assert prediction.diagnostics['candidate']==0


def test_low_stop_confidence_can_explore_without_erasing_unknown_belief(monkeypatch):
    maps=ModuleType('sbfnav.mapdata');maps.GROUND_LEVEL={'map':0.}
    monkeypatch.setitem(sys.modules,'sbfnav.mapdata',maps)
    policy=A3TrajectoryPolicy.__new__(A3TrajectoryPolicy);policy.device=torch.device('cpu')
    policy.record=dict(world_bounds=[0,0,400,400],map_name='map')
    policy.stop_threshold=.3;policy.stop_positive_weight=1.;policy.execution='guarded-direct'
    policy.visited_goal_cells=set();policy.explore_candidates=4
    field=GoalField(torch.tensor([[2.,1.]]),torch.tensor([[0,1]]),torch.tensor([[[.2,.3],[.8,.8]]]),torch.zeros(1,2,16))
    before=field.belief_logits.clone();policy.frozen_field=lambda:(field,{})
    output=TrajectoryOutput(torch.zeros(1,2,3,8,3),torch.zeros(1,2,3),field.belief_logits,torch.tensor([[-5.,5.]]))
    def head(**inputs):
        assert 'target_world' not in inputs and 'target_id' not in inputs
        return output
    policy.head=head
    prediction=policy.predict(SimpleNamespace(pose=np.array([80.,120.,30.,0.])),'near hall')
    assert prediction.diagnostics['candidate']==1 and policy.visited_goal_cells=={0}
    np.testing.assert_allclose(prediction.xyz,[320,320,30])
    assert not prediction.should_stop and torch.equal(before,field.belief_logits)
