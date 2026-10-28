import pytest
from experiments.a2_waldo_landmark_belief.direct_risk_confirmation import verify_new_reserve,CONFIRMATION_CONFIG


def test_next_reserve_cannot_reuse_previously_observed_or_training_targets():
    new=dict(targets=[dict(map_name='m',object_id=1),dict(map_name='m',object_id=2)])
    old=dict(targets=[dict(map_name='m',object_id=3)])
    verify_new_reserve(new,old,{('m',4)},expected=2)
    with pytest.raises(ValueError):verify_new_reserve(new,old,{('m',2)},expected=2)
    with pytest.raises(ValueError):verify_new_reserve(new,dict(targets=[dict(map_name='m',object_id=1)]),set(),expected=2)


def test_new_confirmation_keeps_existing_statistical_risk_requirements():
    risk=CONFIRMATION_CONFIG['risk_criteria']
    assert risk['min_verify_gain_groups']==5 and risk['min_verify_supported_groups']==30
    assert risk['min_verify_a0_correct_groups']==60 and risk['max_verify_broken_group_wilson_upper']==.05
    assert CONFIRMATION_CONFIG['fixed_action']==.25 and CONFIRMATION_CONFIG['no_verify_reselection']
    assert CONFIRMATION_CONFIG['temperatures']==[1.]


def test_committed_confirmation_replays_and_rescores_instead_of_trusting_metadata(tmp_path):
    import numpy as np
    import torch
    from types import SimpleNamespace
    from experiments.a2_waldo_landmark_belief.paths import bootstrap,save,sha
    from experiments.a2_waldo_landmark_belief.direct_risk_confirmation import validate_completed_episode
    bootstrap()
    from graphnav.closedloop import score_trace,ClosedLoopProtocol
    entry=dict(index=2,map_name='m',object_id=7,fold='fit');signature='sig'
    record=SimpleNamespace(start_pose=np.array([0.,0.,50.,0.]),target_position=np.array([1.,1.,50.]),trajectory=np.array([[0.,0.,50.,0.]]))
    raw=dict(signature_sha256=signature,episode=2,step=0,fold='fit',target_instance=['m',7],labels=dict(target_xy=torch.tensor([1.,1.])))
    torch.save(raw,tmp_path/'view.pt')
    poses=[record.start_pose.tolist()];times=[.1]
    info=dict(signature_sha256=signature,entry=entry,poses=poses,termination='policy_stop',decision_seconds=times,
        step_diagnostics=[dict(commanded_xyz=[0.,0.,50.],commanded_stop=True)],
        rows=[dict(filename='view.pt',sha256=sha(tmp_path/'view.pt'),step=0,episode=2,fold='fit')],
        metrics=score_trace(poses,record.target_position,record.trajectory,'policy_stop',decision_seconds=times))
    path=tmp_path/'episode_complete.json';save(info,path)
    def commit():save(dict(complete_sha256=sha(path),signature_sha256=signature,index=2),path.with_suffix('.sha.json'))
    commit();validate_completed_episode(path,tmp_path,entry,signature,record,(-10.,-10.,10.,10.),0.,ClosedLoopProtocol())
    info['metrics']['SR']=0.;save(info,path);commit()
    with pytest.raises(ValueError):validate_completed_episode(path,tmp_path,entry,signature,record,(-10.,-10.,10.,10.),0.,ClosedLoopProtocol())
