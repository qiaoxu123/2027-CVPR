import numpy as np
import torch
from graphnav.object_proposals import box_valid_fraction,select_proposals


def test_box_validity_accounts_for_outside_area_and_unknown_pixels():
    valid=np.ones((16,16),bool)
    boxes=np.array([[.5,.5,.2,.2],[0.,.5,1.,1.],[.5,.5,0.,1.]])
    q=box_valid_fraction(boxes,valid)
    assert q[0]==1 and abs(q[1]-.5)<1e-6 and q[2]==0
    assert not box_valid_fraction(boxes,np.zeros_like(valid)).any()


def test_selection_excludes_invalid_and_deduplicates_without_gt():
    boxes=torch.tensor([[.2,.2,.1,.1],[.2,.2,.1,.1],[.8,.8,.1,.1]])
    scores=torch.tensor([.9,.8,.7]);quality=torch.tensor([1.,1.,0.])
    assert select_proposals(boxes,scores,quality).tolist()==[0]
    quality.zero_();assert not len(select_proposals(boxes,scores,quality))


def test_geometry_prior_can_reorder_valid_queries():
    boxes=torch.tensor([[.2,.2,.1,.1],[.8,.8,.1,.1]])
    scores=torch.tensor([.7,.6]);quality=torch.ones(2)
    assert select_proposals(boxes,scores,quality,budget=1).item()==0
    assert select_proposals(boxes,scores,quality,budget=1,geometry=torch.tensor([-8.,0.])).item()==1
