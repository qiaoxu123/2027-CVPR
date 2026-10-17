from types import SimpleNamespace
import numpy as np
import torch
import pytest
from graphnav import geombelief_control as c
from graphnav.full_localization import GoalField

@pytest.mark.parametrize('arm',['predicted_direct','predicted_waypoint','rule_arrival'])
def test_arrival_not_benchmark_radius(monkeypatch,arm):
    def fake(self,state,field):
        return SimpleNamespace(xyz=np.zeros(3),should_stop=True,diagnostics=dict(selected_goal_xy=[0.,0.],stop_probability=.9,stop_altitude_guard=True))
    monkeypatch.setattr(c,'ORIGINAL_PREDICT_FIELD',fake)
    self=SimpleNamespace(control_arm=arm,execution='guarded-blend',device='cpu')
    field=GoalField(torch.zeros(1,2),torch.tensor([[0]]),torch.zeros(1,1,2),torch.zeros(1,1,128))
    pred=c.controlled_predict_field(self,SimpleNamespace(pose=np.array([1.,0.,20.,0.])),field)
    assert not pred.should_stop and self.execution=='guarded-blend'
    pred=c.controlled_predict_field(self,SimpleNamespace(pose=np.array([0.,0.,20.,0.])),field)
    assert pred.should_stop

def test_rule_removes_only_learned_stop_threshold(monkeypatch):
    def fake(*args):return SimpleNamespace(xyz=np.zeros(3),should_stop=False,diagnostics=dict(selected_goal_xy=[0.,0.],stop_probability=.01,stop_altitude_guard=True))
    monkeypatch.setattr(c,'ORIGINAL_PREDICT_FIELD',fake)
    field=GoalField(torch.zeros(1,2),torch.tensor([[0]]),torch.zeros(1,1,2),torch.zeros(1,1,128))
    self=SimpleNamespace(control_arm='rule_arrival',execution='guarded-blend',device='cpu')
    assert c.controlled_predict_field(self,SimpleNamespace(pose=np.zeros(4)),field).should_stop
