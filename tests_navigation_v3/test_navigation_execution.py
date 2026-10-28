import json
from pathlib import Path
import pytest
from experiments.a2_waldo_landmark_belief import relation_v3_navigation as navigation


def test_pilot_admission_requires_root_bound_candidate_before_runtime(tmp_path,monkeypatch):
    called=[]
    def reject(root):called.append(root);raise ValueError('candidate commit missing')
    monkeypatch.setattr(navigation,'load_candidate',reject)
    with pytest.raises(ValueError,match='commit'):navigation.admit('pilot',tmp_path)
    assert called==[tmp_path]


def test_formal_admission_cannot_use_pilot_only_candidate(tmp_path,monkeypatch):
    called=[]
    def reject(root):called.append(root);raise ValueError('final freeze missing')
    monkeypatch.setattr(navigation,'validate_frozen',reject)
    with pytest.raises(ValueError,match='freeze'):navigation.admit('val_unseen',tmp_path)
    assert called==[tmp_path]


def test_unknown_scope_or_arm_is_rejected_before_path_creation(tmp_path):
    with pytest.raises(ValueError):navigation.arm_directory(tmp_path,'pilot','A1')
    with pytest.raises(ValueError):navigation.arm_directory(tmp_path,'seen128','A2')
    assert not list(tmp_path.iterdir())


def test_actual_run_rejects_candidate_before_device_checks(tmp_path,monkeypatch):
    monkeypatch.setattr(navigation,'BASE',tmp_path)
    monkeypatch.setattr(navigation,'require_cpu_gates',lambda root:None)
    def rejected(root):raise ValueError('candidate commit missing')
    def forbidden(*a,**k):raise AssertionError('Device operation preceded admission')
    monkeypatch.setattr(navigation,'load_candidate',rejected)
    monkeypatch.setattr(navigation,'require_idle_gpu',forbidden)
    with pytest.raises(ValueError,match='commit'):navigation.run('pilot','A2')


def test_synthetic64_complete_rollouts_capture_rescore_and_retain_without_new_inference(tmp_path,monkeypatch):
    """Exercise real runner/rollout/trace code; fake sensor/policy, no actual data or GPU."""
    import numpy as np
    import torch
    from types import SimpleNamespace
    import graphnav.dynamic_runtime as dynamic
    import graphnav.stage1_runtime as runtime
    import sbfnav.dataset as dataset_module
    import sbfnav.planning_state as planning
    from graphnav.geometry import grid_centers
    from graphnav.memory import ObservedRGBMemory
    from sbfnav.mapdata import GROUND_LEVEL
    root=tmp_path/'revision_v3';(root/'training').mkdir(parents=True);(root/'calibration').mkdir()
    for name in ('training/best.pt','navigation_protocol.json','navigation_assets.json','calibration/candidate_calibration.json'):
        (root/name).write_text('{}')
    annotation=tmp_path/'annotations.json';annotation.write_text('{}')
    ended=set();constructed=[];active=[None]
    class Record:
        map_name='synthetic';description_id=0
        def __init__(self,i):self.index=i;self.object_id=f'o{i}';self.instruction=f'query{i}';self.episode_id=[self.map_name,i,0];self.start_pose=np.array([0.,0.,50.,0.])
        @property
        def target_position(self):
            assert self.index in ended,'Private target accessed before whole rollout'
            return np.array([0.,0.,0.])
        @property
        def trajectory(self):
            assert self.index in ended,'Private trajectory accessed before whole rollout'
            return np.array([[0.,0.,50.,0.]])
    class Dataset(list):
        annotation_path=annotation
        def __init__(self,*args):super().__init__([Record(i) for i in range(64)])
    entries=[dict(index=i,map_name='synthetic',object_id=f'o{i}',description_id=0,episode_id=['synthetic',i,0]) for i in range(64)]
    class Catalog:
        def __init__(self,*args):pass
        def referenced_landmarks(self,r):return []
        def referenced_landmark_names(self,r):return []
        def target_object(self,r):assert r.index in ended;return dict(object_type='Car')
    class Builder:
        def __init__(self,*args,**kwargs):pass
        def transform(self,name):return SimpleNamespace(x_min=0.,y_min=0.,x_max=1000.,y_max=1000.)
        def build_from_poses(self,episode,poses):
            active[0]=int(episode.instruction.removeprefix('query'))
            return SimpleNamespace(step=len(poses)-1,pose=poses[-1],current_observation=dict(bgr=np.ones((384,384,3),np.uint8),
                footprint_world=ObservedRGBMemory.view_corners(poses[-1][:2],poses[-1][3],poses[-1][2])))
    class Model(torch.nn.Module):
        def forward(self):
            field=SimpleNamespace(belief_logits=torch.arange(784).float()[None]);return SimpleNamespace(static=field,belief=field)
    class Policy:
        def __init__(self,*args):
            constructed.append(True);self.model=Model();self.args=SimpleNamespace(grid_size=28)
            self.stop_threshold=.3;self.execution='guarded-blend';self.explore_candidates=0
            self.world=(grid_centers(28)*1000)[None]
        def set_episode(self,public,*args):
            assert not hasattr(public,'target_position') and not hasattr(public,'trajectory')
        def predict(self,state,instruction):
            out=self.model();points=self.world[0,runtime.retrieval_ids(out.belief.belief_logits)[0]].tolist()
            return SimpleNamespace(xyz=np.array([0.,0.,50.]),should_stop=True,diagnostics=dict(observation_step=state.step,
                static_top16_world_xy=points,belief_top16_world_xy=points))
    original_rollout=navigation.rollout
    def checked_rollout(*args,**kwargs):
        result=original_rollout(*args,**kwargs);ended.add(active[0]);return result
    monkeypatch.setattr(navigation,'BASE',tmp_path);monkeypatch.setattr(navigation,'require_cpu_gates',lambda r:None)
    monkeypatch.setattr(navigation,'admit',lambda s,r:(root/'calibration/candidate_calibration.json',None))
    monkeypatch.setattr(navigation,'require_assets',lambda *a,**k:None)
    monkeypatch.setattr(navigation,'require_idle_gpu',lambda:None)
    monkeypatch.setattr(navigation,'source_entries',lambda *a,**k:entries)
    monkeypatch.setattr(navigation,'rollout',checked_rollout)
    monkeypatch.setattr(dynamic,'NativeObservationBuilder',Builder);monkeypatch.setattr(planning,'PlanningStateBuilder',Builder)
    monkeypatch.setattr(dataset_module,'CityNavDataset',Dataset);monkeypatch.setattr(dataset_module,'CityReferCatalog',Catalog)
    monkeypatch.setattr(runtime,'GraphNavStage1Policy',Policy);monkeypatch.setitem(GROUND_LEVEL,'synthetic',0.)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None);monkeypatch.setattr(torch,'manual_seed',lambda x:None)
    monkeypatch.setattr(torch.backends.cuda.matmul,'allow_tf32',torch.backends.cuda.matmul.allow_tf32)
    monkeypatch.setattr(torch.backends.cudnn,'allow_tf32',torch.backends.cudnn.allow_tf32)
    navigation.run('pilot','A0')
    directory=root/'pilot/A0';complete=json.loads((directory/'complete.json').read_text())
    assert complete['episodes']==64 and len(ended)==64 and len(constructed)==1
    before=(directory/'metrics.json').read_bytes()
    def forbidden():raise AssertionError('Completed run needlessly reacquired GPU')
    monkeypatch.setattr(navigation,'require_idle_gpu',forbidden)
    navigation.run('pilot','A0')
    assert (directory/'metrics.json').read_bytes()==before and len(constructed)==1
