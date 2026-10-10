import numpy as np
import pytest
from experiments.visual_goal_association.association import associate,fuse,identity_oracle,grid_geometry

GRID=np.stack(np.meshgrid((np.arange(28)+.5)/28,(np.arange(28)+.5)/28),-1).reshape(-1,2)
BOUNDS=[0,0,28,28]
VIEW=[[0,0],[28,0],[28,28],[0,28]]

@pytest.mark.parametrize('method',['centroid','polygon_overlap','distance_soft','instance'])
def test_observed_tiny_object_and_unknown_exact(method):
 w=associate([[2.1,2.1],[2.4,2.1],[2.4,2.4],[2.1,2.4]],VIEW,BOUNDS,GRID,method)
 assert w[2*28+2]>0 and w.max()<=1
 logits=np.linspace(-2,2,784)
 u,k,r=fuse(logits,w[None],[2],gain=4,method=method)
 assert np.array_equal(u[~k],logits[~k]) and (r[~k]==0).all()

@pytest.mark.parametrize('method',['centroid','polygon_overlap','distance_soft','instance'])
def test_unobserved_object_has_no_evidence(method):
 w=associate([[25,25],[27,25],[27,27],[25,27]],[[0,0],[10,0],[10,10],[0,10]],BOUNDS,GRID,method)
 assert not w.any()

def test_large_irregular_polygon_centroid_is_not_area():
 p=[[1,1],[9,1],[9,2],[2,2],[2,9],[1,9]]
 overlap=associate(p,VIEW,BOUNDS,GRID,'polygon_overlap')
 assert overlap[1*28+7]>0 and overlap[7*28+1]>0
 assert overlap[5*28+5]==0
 assert (overlap>0).sum()>1

def test_footprint_clipping_prevents_future_surface_evidence():
 p=[[1,1],[20,1],[20,20],[1,20]]
 w=associate(p,[[0,0],[5,0],[5,5],[0,5]],BOUNDS,GRID,'polygon_overlap')
 assert w[2*28+2]>0 and w[10*28+10]==0

def test_collision_preserves_stronger_instance():
 w=np.zeros((2,784));w[:,3]=1
 _,_,mean=fuse(np.zeros(784),w,[4,-4],gain=1,method='polygon_overlap')
 _,_,maximum=fuse(np.zeros(784),w,[4,-4],gain=1,method='instance')
 assert mean[3]==0 and maximum[3]==4

def test_oracle_uses_static_within_instance_not_target_position():
 w=np.zeros((1,784));w[0,[2,3]]=1
 logits=np.zeros(784);logits[1]=10;logits[3]=4
 u,k,r=identity_oracle(logits,w,[True])
 assert u.argmax()==3 and (r[~k]==0).all()

def test_invalid_geometry_rejected():
 with pytest.raises(ValueError):associate([[0,0],[1,1],[1,0],[0,1]],VIEW,BOUNDS,GRID,'instance')

@pytest.mark.parametrize('strategy',['hold','approach','descend','lateral','belief_active'])
def test_observation_policy_public_and_bounded(strategy):
 from experiments.visual_goal_association.coverage import public_move
 pose=np.array([14,14,40,0.]);bounds=np.array(BOUNDS);logits=np.linspace(-2,2,784)
 decision=public_move(pose,bounds,0,GRID,logits,np.zeros(784),'a red vehicle',strategy,0)
 assert decision is not None
 assert np.linalg.norm(decision['pose'][:2]-pose[:2])<=50+1e-8
 assert abs(decision['pose'][2]-pose[2])<=10
 assert (decision['pose'][:2]>=bounds[:2]).all() and (decision['pose'][:2]<=bounds[2:]).all()

def test_invalid_altitude_no_invented_flight():
 from experiments.visual_goal_association.coverage import public_move
 assert public_move(np.array([14,14,200,0.]),np.array(BOUNDS),0,GRID,np.zeros(784),np.zeros(784),'building','approach',0) is None

def test_instance_history_does_not_use_future_scores():
 from experiments.visual_goal_association.evaluate import observed_candidates
 weights={m:np.eye(1,784)[0] for m in ['centroid','polygon_overlap','distance_soft','instance']}
 def reg(step,score):return dict(object_id='public-instance',object_type='Building',quality=1,step=step,weights=weights,scores={'real':score},confidences={'real':.5})
 states=[dict(episode_id='episode',cutoff_step=0,regions=[reg(0,1)]),dict(episode_id='episode',cutoff_step=5,regions=[reg(5,9)])]
 result=list(observed_candidates(states,True,'real'))
 assert result[0][1][0]['score']==1 and result[0][1][0]['views']==1
 assert result[1][1][0]['score']==5 and result[1][1][0]['views']==2
 bad=[dict(episode_id='episode',cutoff_step=0,regions=[reg(5,9)])]
 with pytest.raises(AssertionError):list(observed_candidates(bad,True,'real'))
