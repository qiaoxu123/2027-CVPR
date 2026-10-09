import numpy as np
import torch
from graphnav.dynamic_runtime import contour_matches
from graphnav.trajectory_policy import convergent_first_waypoint
from types import SimpleNamespace
import json


def test_public_contour_binding_rejects_homonymous_different_fragment():
    contour=np.array([[0,0],[10,0],[10,10],[0,10]],float)
    assert contour_matches(contour,np.concatenate((contour,contour[:1])))
    assert not contour_matches(contour,contour+20)


def test_path_taper_retains_learned_deviation_and_vanishes_at_predicted_goal():
    pose=torch.tensor([[0.,0.,30.,0.]])
    waypoint,distance=convergent_first_waypoint(pose,torch.tensor([[100.,0.]]),torch.tensor([[40.,20.,30.]]),blend=.25)
    assert torch.equal(waypoint,torch.tensor([[47.5,5.,30.]]))
    assert distance.item()==100
    arrived=torch.tensor([[100.,0.,30.,0.]])
    waypoint,distance=convergent_first_waypoint(arrived,torch.tensor([[100.,0.]]),torch.tensor([[110.,20.,30.]]),blend=1.)
    assert torch.equal(waypoint,torch.tensor([[100.,0.,30.]])) and distance.item()==0


def test_online_history_uses_only_observed_vectors_and_zero_gate_preserves_a3(tmp_path):
    from graphnav.dynamic_runtime import DynamicWaypointPolicy
    from graphnav.dynamic_belief import DynamicGoalBelief
    from graphnav.full_localization import FullTokenGeometryLocalizer
    from graphnav.candidate_geometry import FullCandidateDataset,collate_full
    record=dict(map_name='map',instruction='east of hall',world_bounds=[0,0,400,400],
        physical_map_sizes=[400,400],target_xy=[0,0],landmarks=[dict(name='hall',
        object_type='Building',contour=[[.2,.2],[.3,.2],[.3,.3],[.2,.3]])])
    path=tmp_path/'query.jsonl';path.write_text(json.dumps(record)+'\n')
    dataset=FullCandidateDataset(path,grid_size=4,max_landmarks=2,max_roads=0,max_relations=2)
    batch=collate_full([dataset[0]])
    a3=FullTokenGeometryLocalizer(hidden=16,text_dim=12,heads=4,max_relations=2,top_k=3,dropout=0).eval()
    names=torch.randn(1,4,12)
    base=a3(batch,variant='geometry',tokens=torch.randn(1,5,12),token_mask=torch.ones(1,5,dtype=torch.bool),node_emb=names)
    runtime=DynamicWaypointPolicy.__new__(DynamicWaypointPolicy)
    runtime.device=torch.device('cpu');runtime.record=record;runtime.public_node_emb=names
    runtime.public_entities=dataset._prepared(0)[0]['entities'];runtime.frozen_field=lambda:(base,batch)
    runtime.dynamic=DynamicGoalBelief(a3,vision_dim=256).eval()
    runtime.use_graph=True;runtime.use_visual=True;runtime.use_history=True
    runtime.evidence={};runtime.geometry={};runtime.pose_history=[];runtime.landmarks=[];runtime.observed_cells=None
    roi=SimpleNamespace(object_id='public-region',name='hall',contour_world=np.array([[80,80],[120,80],[120,120],[80,120]]),
        confidence=.5,quality={'visible_pixels':300})
    class Encoder:
        def observe(self,observation,landmarks,*,step):
            assert observation['actual_step']==step
            return ([roi],torch.full((1,256),float(1+2*step))) if step<2 else ([],torch.zeros(0,256))
    runtime.encoder_visual=Encoder()
    for step in range(3):
        observation=dict(actual_step=step,bgr=np.full((384,384,3),80,dtype=np.uint8),footprint_world=np.array([[0,400],[400,400],[400,0],[0,0]]))
        result=runtime.predict(SimpleNamespace(pose=np.array([10.,10.,30.,0.]),step=step,current_observation=observation),record['instruction'])
        assert result.diagnostics['candidate']==int(base.belief_logits.argmax())
        assert result.diagnostics['observed_nodes']==1
    item=runtime.evidence['public-region']
    assert item['confidence']==.5 and item['mass']==1.
    assert torch.equal(item['vector']/item['mass'],torch.full((256,),2.))


def test_observed_coverage_never_marks_outside_or_black_cells_as_known():
    from graphnav.dynamic_runtime import observed_grid_cells
    bgr=np.full((384,384,3),80,dtype=np.uint8);bgr[:200,:200]=0
    observation=dict(bgr=bgr,footprint_world=np.array([[0,400],[400,400],[400,0],[0,0]]))
    points=np.array([[50,350],[350,350],[500,500]])
    assert observed_grid_cells(points,observation).tolist()==[False,True,False]
