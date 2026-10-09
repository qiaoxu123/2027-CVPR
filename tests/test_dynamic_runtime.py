import numpy as np
import torch
from graphnav.dynamic_runtime import contour_matches
from graphnav.trajectory_policy import convergent_first_waypoint


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
