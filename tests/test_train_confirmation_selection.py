import pytest
from graphnav.train_confirmation_selection import selected_train_tasks


def test_selection_uses_frozen_identity_and_never_target_position():
    raw=[dict(area='m',block=1,object_ids=[7],target_positions=[[1,2,3]])]
    catalog={'m_block_1':{'7':{'object_type':'Car'}}}
    selected=[dict(index=0,physical_group=['m_block_1',7],category='Car')]
    assert selected_train_tasks(raw,catalog,selected)==[(('Car','m_block_1',7),0)]
    raw[0]['target_positions']=[[999,999,999]]
    assert selected_train_tasks(raw,catalog,selected)==[(('Car','m_block_1',7),0)]


def test_duplicate_physical_target_and_mismatched_manifest_are_rejected():
    raw=[dict(area='m',block=1,object_ids=[7])]
    catalog={'m_block_1':{'7':{'object_type':'Car'}}}
    row=dict(index=0,physical_group=['m_block_1',7],category='Car')
    with pytest.raises(ValueError):selected_train_tasks(raw,catalog,[row,row])
    with pytest.raises(ValueError):selected_train_tasks(raw,catalog,[dict(row,physical_group=['m_block_1',8])])
