import torch
from graphnav.native_arrival import arrival_features,FEATURE_NAMES
from scripts.train_native_arrival_probe import public_pose


def test_onpolicy_pose_never_reads_teacher_future_or_private_record():
    class Forbidden:
        def __getitem__(self,key):raise AssertionError('Teacher record must not be read for actual rollout pose')
    row=dict(pose=torch.tensor([1.,2.,30.]),episode=0,step=100)
    assert torch.equal(public_pose(row,Forbidden()),row['pose'])


def test_arrival_summaries_use_public_world_distances_and_respect_padding():
    q=torch.tensor([[8.,100.]]);null=torch.tensor([-8.]);world=torch.tensor([[[0.,0.],[50.,50.]]]);valid=torch.tensor([[True,False]]);geometry=torch.zeros(1,2,8);geometry[:,:,2]=1
    base=torch.zeros(1,3);grid=torch.tensor([[[0.,0.],[10.,0.],[50.,0.]]]);pose=torch.zeros(1,3)
    x=arrival_features(q,null,world,valid,geometry,base,base,grid,pose)
    assert x.shape==(1,len(FEATURE_NAMES)) and torch.isfinite(x).all()
    assert x[0,0]>.99 and x[0,12]>.99
    distant=arrival_features(q,null,world,valid,geometry,base,base,grid,torch.tensor([[80.,0.,0.]]))
    assert distant[0,0]==0 and distant[0,5]==0
