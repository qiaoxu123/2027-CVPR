import numpy as np
import torch
from graphnav.trajectory_policy import CandidateTrajectoryStop,TrajectoryOutput,remaining_waypoints,trajectory_losses


def test_supervision_uses_actual_curved_human_remainder():
    human=np.array([[0,0,30],[50,0,40],[50,50,50],[100,50,50]])
    actual=remaining_waypoints(human,0,count=4,spacing_m=50)
    np.testing.assert_allclose(actual,[[50,0,40],[50,50,50],[100,50,50],[100,50,50]])
    np.testing.assert_allclose(remaining_waypoints(human,3,count=2),[human[-1],human[-1]])


def test_selection_matches_candidate_mode_and_stop_identity():
    paths=torch.arange(2*3*2*3).reshape(1,2,3,2,3).float()
    output=TrajectoryOutput(paths,torch.tensor([[[0.,0.,0.],[0.,4.,0.]]]),
        torch.tensor([[1.,2.]]),torch.tensor([[-3.,7.]]))
    candidate,mode,waypoint,stop=output.select()
    assert candidate.item()==1 and mode.item()==1
    assert torch.equal(waypoint[0],paths[0,1,1,0]) and stop.item()==7.


def test_head_executes_bounded_first_step_and_label_only_losses_backward():
    net=CandidateTrajectoryStop(hidden=16,modes=3,waypoints=4)
    inputs=dict(candidate_xy=torch.tensor([[[.9,.8],[.2,.3]]]),
        candidate_context=torch.randn(1,2,16),candidate_logits=torch.tensor([[2.,1.]]),
        pose_world=torch.tensor([[40.,40.,30.,.5]]),bounds=torch.tensor([[0.,0.,400.,400.]]),
        ground_level=torch.tensor([0.]),belief_entropy=torch.tensor([.5]),observed_fraction=torch.tensor([.3]))
    with torch.no_grad():net.path.weight.normal_()
    out=net(**inputs)
    start=inputs['pose_world'][:,None,None,None,:3].expand(-1,2,3,1,-1)
    segments=torch.diff(torch.cat((start,out.paths_world),3),dim=3)
    assert segments[...,:2].norm(dim=-1).max()<=50.0001
    assert segments[...,2].abs().max()<=10.0001
    assert out.paths_world[...,:2].min()>=0 and out.paths_world[...,:2].max()<=400
    losses=trajectory_losses(out,candidate_world_xy=inputs['candidate_xy']*400,
        pose_world=inputs['pose_world'],target_world=torch.tensor([[80.,120.,30.]]),
        teacher_world=torch.tensor([[[60.,60.,30.],[70.,80.,30.],[80.,100.,30.],[80.,120.,30.]]]))
    assert torch.isfinite(losses['total'])
    losses['total'].backward();assert net.path.weight.grad.abs().sum()>0


def test_illegal_teacher_altitude_is_masked_without_fabricating_waypoints():
    paths=torch.zeros(1,1,1,2,3,requires_grad=True)
    out=TrajectoryOutput(paths,torch.zeros(1,1,1),torch.zeros(1,1),torch.zeros(1,1))
    kwargs=dict(candidate_world_xy=torch.zeros(1,1,2),pose_world=torch.zeros(1,4),
        target_world=torch.zeros(1,3),valid_teacher_z=torch.zeros(1,2,dtype=torch.bool))
    first=trajectory_losses(out,teacher_world=torch.zeros(1,2,3),**kwargs)
    teacher=torch.zeros(1,2,3);teacher[...,2]=1000
    second=trajectory_losses(out,teacher_world=teacher,**kwargs)
    assert torch.equal(first['path'],second['path'])


def test_conditional_mode_score_cannot_arbitrarily_override_candidate_rank():
    paths=torch.zeros(1,2,3,1,3)
    output=TrajectoryOutput(paths,torch.tensor([[[0.,0.,0.],[100.,100.,100.]]]),
        torch.tensor([[2.,1.]]),torch.zeros(1,2))
    assert output.select()[0].item()==0


def test_initial_multimodal_paths_are_distinct_without_future_labels():
    model=CandidateTrajectoryStop(hidden=16)
    out=model(candidate_xy=torch.tensor([[[.9,.9]]]),candidate_context=torch.zeros(1,1,16),
        candidate_logits=torch.zeros(1,1),pose_world=torch.tensor([[10.,10.,30.,0.]]),
        bounds=torch.tensor([[0.,0.,400.,400.]]),ground_level=torch.zeros(1),
        belief_entropy=torch.ones(1),observed_fraction=torch.zeros(1))
    assert not torch.equal(out.paths_world[:,:,0],out.paths_world[:,:,1])
    assert not torch.equal(out.paths_world[:,:,1],out.paths_world[:,:,2])
