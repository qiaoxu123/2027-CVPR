from types import SimpleNamespace
import torch
from graphnav.stop_recovery import CandidateStopRecoveryDataset


def base(target):
    return SimpleNamespace(features={'candidate_xy':torch.tensor([[[.1,.1],[.8,.8]]]),
        'bounds':torch.tensor([[0.,0.,100.,100.]]),'candidate_context':torch.zeros(1,2,4),
        'candidate_logits':torch.zeros(1,2),'belief_entropy':torch.ones(1)},rows=[dict(query_index=0,
        source_episode_index=12,initial_pose_world=[0.,0.,50.,0.],ground_level=0.,labels={'target_world':target})])


def test_recovery_poses_are_label_independent_and_labels_match_actual_success_radius():
    a=CandidateStopRecoveryDataset(base([10.,10.,0.]),candidates=2)
    b=CandidateStopRecoveryDataset(base([80.,80.,0.]),candidates=2)
    for i in range(len(a)):
        x,y=a[i];other,z=b[i]
        assert all(torch.equal(x[k],other[k]) for k in x)
        assert 'target_world' not in x and 'stop' not in x
    assert a[0][1]['stop'].item()==1 and b[0][1]['stop'].item()==0
    assert a[2][1]['stop'].item()==0 and b[2][1]['stop'].item()==1
