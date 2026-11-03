import pytest
import torch
from graphnav.target_language_conditioning import condition_target_language


def test_full_instruction_retains_reference_and_original_appearance_metadata():
    row=dict(target_text=torch.tensor([1.,0.]),full_text=torch.tensor([0.,1.]),
             context_text=torch.tensor([2.,3.]),prior_logits=torch.tensor([4.,5.]),
             target_phrase='black car',target_world=torch.tensor([20.,30.]))
    out=condition_target_language([row],'full-instruction')[0]
    assert torch.equal(out['target_text'],row['full_text'])
    assert out['context_text'] is row['context_text']
    assert out['prior_logits'] is row['prior_logits']
    assert out['target_phrase']=='black car'
    assert torch.equal(row['target_text'],torch.tensor([1.,0.]))


def test_private_labels_do_not_select_or_modify_text_features():
    public=dict(target_text=torch.tensor([1.,0.]),full_text=torch.tensor([0.,1.]),context_text=torch.tensor([2.,3.]))
    a=condition_target_language([dict(public,target_world=[0,0],instance_id=1)],'full-instruction')[0]
    b=condition_target_language([dict(public,target_world=[999,999],instance_id=8)],'full-instruction')[0]
    for key in public:assert torch.equal(a[key],b[key])


def test_role_default_preserves_tensor_values_and_rejects_unknown_modes():
    row=dict(target_text=torch.tensor([1.,0.]))
    assert condition_target_language([row],'role')[0]['target_text'] is row['target_text']
    with pytest.raises(ValueError):condition_target_language([row],'guess')
    with pytest.raises(ValueError):condition_target_language([row],'full-instruction')
