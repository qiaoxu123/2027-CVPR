import pytest
from scripts.cache_local_chain_training import assign_maps


def test_new_fit_observations_retain_frozen_map_partition():
    original={'a':'fit','b':'dev','c':'confirm'}
    assert assign_maps(['a'],original)=={'a':'fit'}
    with pytest.raises(ValueError,match='missing'):
        assign_maps(['unseen'],original)
    with pytest.raises(ValueError,match='invalid'):
        assign_maps(['a'],{'a':'training'})


def test_small_new_cache_cannot_silently_create_new_holdouts():
    with pytest.raises(ValueError,match='nine'):
        assign_maps(['a','b'])


def test_onpolicy_augmentation_cannot_import_dev_targets_or_repeat_frames():
    from scripts.train_local_chain_position import merge_fit_observations
    base=[dict(task_key='fit1',frame_key='original')]
    extra=[dict(task_key='fit1',frame_key='onpolicy')]
    assert len(merge_fit_observations(base,extra))==2
    with pytest.raises(ValueError,match='non-fit'):
        merge_fit_observations(base,[dict(task_key='dev1',frame_key='new')])
    with pytest.raises(ValueError,match='duplicate'):
        merge_fit_observations(base,[dict(task_key='fit1',frame_key='original')])
