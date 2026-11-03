import torch
import pytest
from graphnav.retarget_training_data import select_training_query, weak_instance_mask, retarget_overrides


def test_query_selection_is_stable_and_not_based_on_outcomes():
    assert select_training_query([9,3,6], 'm', 'a') == select_training_query([6,9,3], 'm', 'a')
    assert select_training_query([7], 'm', 'a') == 7


def test_private_core_mask_preserves_unknown_and_rejects_conflicts():
    mask=weak_instance_mask({'a':[1], 'b':[3]},'a',5)
    assert mask.tolist()==[-1,1,-1,0,-1]
    with pytest.raises(ValueError):weak_instance_mask({'a':[1], 'b':[1]},'a',5)


def test_all_language_conditioned_fields_are_overridden():
    source=dict(prior_logits=torch.zeros(4),target_text=torch.zeros(8),context_text=torch.zeros(8),
                public_geometry=torch.zeros(2,31),image_features=torch.randn(2,3,8),
                local_uv=torch.rand(2,2),quality=torch.ones(2,6),local_valid=torch.ones(2,dtype=torch.bool))
    source.update(target_world=torch.ones(2),instance_id='private-old-target',full_text=torch.zeros(8))
    query=dict(prior_logits=torch.ones(4),target_text=torch.ones(8),context_text=torch.ones(8),
               full_text=torch.ones(8)*2,target_phrase='the white car')
    geometry=torch.ones(2,31)
    output=retarget_overrides(source,query,geometry)
    assert output['prior_logits'].sum()==4 and output['context_text'].sum()==8
    assert output['public_geometry'].sum()==62
    assert output['full_text'].sum()==16
    assert output['image_features'].data_ptr()==source['image_features'].data_ptr()
    assert 'target_world' not in output and 'instance_id' not in output
