import torch
import pytest
from experiments.a2_waldo_landmark_belief.revision_controls import controlled_inputs


def test_text_suppression_removes_both_sources_without_mutating_original():
    original=dict(instruction_tokens=torch.ones(1,2,3),landmark_name_tokens=torch.ones(1,1,2,3))
    altered=controlled_inputs(original,{},'no_text')
    assert not altered['instruction_tokens'].any() and not altered['landmark_name_tokens'].any()
    assert original['instruction_tokens'].all() and altered['control']=='no_text'


def test_image_relative_direction_uses_actual_observation_basis():
    geometry=torch.zeros(1,1,1,14);geometry[...,0]=1;geometry[...,4]=1;geometry[...,2]=3
    network=dict(pair_geometry=geometry)
    meta=dict(footprint_world=[[0,1],[0,2],[1,2],[1,1]])
    result=controlled_inputs(network,meta,'image_relative')['pair_geometry']
    assert result[...,0].item()==0 and result[...,1].item()==1
    assert result[...,3].item()==1 and result[...,4].item()==0
    assert result[...,2].item()==3 and geometry[...,0].item()==1


def test_failed_detection_masks_candidates_and_invalid_controls_rejected():
    network=dict(box_valid=torch.ones(1,2,dtype=torch.bool),pair_valid=torch.ones(1,2,3,dtype=torch.bool))
    result=controlled_inputs(network,{},'failed_detection')
    assert not result['box_valid'].any() and not result['pair_valid'].any()
    with pytest.raises(ValueError):controlled_inputs(network,{},'unknown')
