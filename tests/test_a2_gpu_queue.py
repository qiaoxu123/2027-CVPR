import pytest
from experiments.a2_waldo_landmark_belief.wait_cache_then_train import cache_ready


def test_active_cache_with_zero_exit_status_does_not_authorize_training():
    assert not cache_ready(dict(LoadState='loaded',ActiveState='active',ExecMainStatus='0',Result='success'))
    assert cache_ready(dict(LoadState='loaded',ActiveState='inactive',ExecMainStatus='0',Result='success'))


def test_missing_or_failed_cache_requires_inspection_instead_of_restarting():
    with pytest.raises(RuntimeError):cache_ready(dict(LoadState='not-found',ActiveState='inactive',ExecMainStatus='0',Result='success'))
    with pytest.raises(RuntimeError):cache_ready(dict(LoadState='loaded',ActiveState='failed',ExecMainStatus='1',Result='exit-code'))


def test_child_keeps_gpu_lock_after_parent_descriptor_closes(tmp_path):
    import fcntl,subprocess,sys
    from experiments.a2_waldo_landmark_belief.wait_cache_then_train import spawn_locked
    path=tmp_path/'gpu.lock'
    lock=path.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    child=spawn_locked([sys.executable,'-c',"import time; print('ready',flush=True); time.sleep(5)"],
                       lock.fileno(),stdout=subprocess.PIPE,text=True)
    try:
        assert child.stdout.readline().strip()=='ready'
        lock.close()
        with path.open('a') as other:
            with pytest.raises(BlockingIOError):fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
    finally:
        lock.close();child.terminate();child.wait(timeout=5)


def test_refinement_smoke_requires_reachable_offset_supervision():
    import torch
    from experiments.a2_waldo_landmark_belief.verify_rows import informative_refinement_view
    row=dict(labels=dict(ranking=torch.tensor([1.,0]),rank_label_mask=torch.tensor([True,True]),
                         offset_valid=torch.tensor([False,False])),
             public=dict(network=dict(link_valid=torch.tensor([True]))))
    assert not informative_refinement_view(row)
    row['labels']['offset_valid'][0]=True
    assert informative_refinement_view(row)
