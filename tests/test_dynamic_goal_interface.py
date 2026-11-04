import torch
from graphnav.full_localization import GoalField


def make_field(order):
 uv=torch.tensor([[[.2,.2],[.4,.4],[.6,.6],[.8,.8]]]);ids=torch.tensor([order]);dense=torch.arange(24.).view(1,4,6);logits=torch.tensor([[.1,.4,.3,.2]])
 return GoalField(logits,ids,uv.gather(1,ids[...,None].expand(-1,-1,2)),dense.gather(1,ids[...,None].expand(-1,-1,6)),dense_context=dense),uv


def test_new_goal_requires_confirmation_and_preserves_context():
 from graphnav.dynamic_goal_interface import CandidateCommitment
 c=CandidateCommitment();f,uv=make_field([0,1,2]);c.update(f,uv)
 g,uv=make_field([1,0,2]);held,a=c.update(g,uv)
 assert held.candidate_ids[0,0]==0 and a['pending']
 assert torch.equal(held.belief_logits,g.belief_logits)
 assert torch.equal(held.candidate_context,g.dense_context.gather(1,held.candidate_ids[...,None].expand(-1,-1,6)))
 accepted,b=c.update(g,uv);assert accepted.candidate_ids[0,0]==1 and not b['pending']


def test_wrong_goal_can_be_replaced_without_unlimited_lock():
 from graphnav.dynamic_goal_interface import CandidateCommitment
 c=CandidateCommitment();f,uv=make_field([0,1,2]);c.update(f,uv)
 g,uv=make_field([1,0,2]);c.update(g,uv)
 h,uv=make_field([2,0,1]);updated,a=c.update(h,uv)
 assert updated.candidate_ids[0,0]==2 and not a['pending']
 c=CandidateCommitment();c.update(f,uv);h,uv=make_field([2,1,3]);updated,a=c.update(h,uv)
 assert updated.candidate_ids[0,0]==2 and not a['pending']


def test_arrival_depends_on_public_grid_and_learned_stop_not_gt():
 from graphnav.dynamic_goal_interface import grid_arrival_tolerance,interface_stop
 tol=grid_arrival_tolerance([0,0,400,100],28)
 assert abs(tol-((400/28)**2+(100/28)**2)**.5/2)<1e-6
 assert interface_stop(probability=.4,distance=3.,altitude_guard=True,pending=False,tolerance=tol)
 assert not interface_stop(probability=.2,distance=0.,altitude_guard=True,pending=False,tolerance=tol)
 assert not interface_stop(probability=.9,distance=0.,altitude_guard=True,pending=True,tolerance=tol)
 assert not interface_stop(probability=.9,distance=0.,altitude_guard=False,pending=False,tolerance=tol)
 assert grid_arrival_tolerance([0,0,10000,10000],28)==20.
