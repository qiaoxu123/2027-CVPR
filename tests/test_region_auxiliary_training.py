import torch
import pytest
from scripts.train_region_auxiliary_goal import collate, paired_changes
from scripts.train_local_chain_position import batch_inputs, INPUTS


def test_private_region_labels_never_enter_normal_forward_inputs():
    row = {key: torch.zeros(1) for key in INPUTS}
    row.update(candidate_world=torch.ones(2, 2), target_world=torch.zeros(2),
               frame_key='x', task_key='t', category='Building', private_region_labels=torch.tensor([1]))
    batch = collate([row])
    public = batch_inputs(batch, 'cpu')
    assert 'private_region_labels' in batch and 'private_region_labels' not in public
    assert 'target_world' not in public and 'candidate_world' not in public


def test_paired_changes_require_same_frames_and_do_not_depend_on_row_order():
    before = {'rows': [{'frame_key': 'a', 'success': True}, {'frame_key': 'b', 'success': False}]}
    after = {'rows': [{'frame_key': 'b', 'success': True}, {'frame_key': 'a', 'success': False}]}
    assert paired_changes(after, before) == {'corrections': 1, 'harms': 1, 'net': 0}
    with pytest.raises(ValueError): paired_changes({'rows': []}, before)
