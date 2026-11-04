import inspect
import torch
from graphnav.landmark_visual_only import visual_only_world_belief
from graphnav.geometry_guided_visual import visual_world_beliefs


def test_visual_only_signature_rejects_geometry_and_polygon():
    assert set(inspect.signature(visual_only_world_belief).parameters)=={'local_uv','scores','valid','grid_size'}


def test_visual_only_matches_previous_raw_without_reading_geometry():
    torch.manual_seed(9)
    uv=torch.rand(2,16,2);scores=torch.randn(2,16);valid=torch.ones(2,16,dtype=torch.bool)
    actual=visual_only_world_belief(local_uv=uv,scores=scores,valid=valid)
    for geometry in (torch.randn(2,784),torch.randn(2,784)*100):
        old=visual_world_beliefs(geometry,uv,scores,valid,torch.ones(2,dtype=torch.bool))
        assert torch.equal(actual['covered'],old['covered'])
        assert torch.allclose(actual['probabilities'],old['raw'].softmax(-1))
    assert torch.allclose(actual['probabilities'].sum(-1),torch.ones(2))


def test_empty_view_abstains_and_outside_view_has_no_mass():
    uv=torch.tensor([[[.125,.125],[.875,.875]]]*2);score=torch.tensor([[0.,3.],[0.,3.]])
    valid=torch.tensor([[True,True],[False,False]])
    result=visual_only_world_belief(local_uv=uv,scores=score,valid=valid,grid_size=4)
    assert result['has_view'].tolist()==[True,False]
    assert result['probabilities'][1].sum()==0
    assert result['probabilities'][~result['covered']].sum()==0
    assert result['probabilities'][0].argmax()==15


def test_coordinate_mapping_changes_world_position_without_changing_image_score():
    score=torch.tensor([[2.]])
    first=visual_only_world_belief(local_uv=torch.tensor([[[.125,.125]]]),scores=score,valid=torch.ones(1,1,dtype=torch.bool),grid_size=4)
    second=visual_only_world_belief(local_uv=torch.tensor([[[.875,.875]]]),scores=score,valid=torch.ones(1,1,dtype=torch.bool),grid_size=4)
    assert first['probabilities'].argmax()==0 and second['probabilities'].argmax()==15


def test_uniform_map_tie_break_is_not_changed_by_float_splat_noise():
    import numpy as np
    from scripts.audit_visual_only_landmarks import prediction_index
    world=np.array([[0.,0.],[1.,0.],[2.,0.]])
    covered=np.ones(3,dtype=bool)
    assert prediction_index(np.array([1/3,1/3,1/3]),covered,world)==1
    assert prediction_index(np.array([1/3+1e-8,1/3,1/3]),covered,world)==1
