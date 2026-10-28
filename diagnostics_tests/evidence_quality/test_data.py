"""Public forward stays independent from private offline outcome calculation."""
from copy import deepcopy
import importlib

import pytest
import torch

from experiments.a2_waldo_landmark_belief.relation_v3_model import RelationV3Network


def api():
    return importlib.import_module('diagnostics.evidence_quality.data')


def example():
    torch.manual_seed(0)
    model = RelationV3Network(text_dim=8, vision_dim=6, hidden=8, heads=2).eval()
    network = dict(instruction_tokens=torch.randn(3,8), token_valid=torch.ones(3,dtype=torch.bool),
        mention_mask=torch.tensor([[False,True,False]]), landmark_name_tokens=torch.randn(1,2,8),
        name_valid=torch.ones(1,2,dtype=torch.bool), link_valid=torch.ones(1,dtype=torch.bool),
        box_rgb=torch.randn(2,6), box_valid=torch.ones(2,dtype=torch.bool),
        pair_geometry=torch.zeros(2,1,14), pair_valid=torch.ones(2,1,dtype=torch.bool),
        offset_limit=torch.ones(2,2), box_kind=torch.zeros(2,dtype=torch.long))
    spatial = dict(base_logits=torch.tensor([4.,0.]), grid_world=torch.tensor([[0.,0.],[40.,0.]]),
        candidate_xy=torch.tensor([[0.,0.],[40.,0.]]),
        candidate_support=torch.tensor([[True,False],[False,True]]))
    return model, dict(network=network, spatial=spatial)


def test_private_outcomes_cannot_change_public_features_or_actions():
    model, public = example(); result = api().public_forward(model, public)
    frozen = {k:v.clone() for k,v in result.items()}
    a = api().private_outcomes(result, torch.tensor([0.,0.]))
    b = api().private_outcomes(result, torch.tensor([40.,0.]))
    assert a['before'] == 0 and b['before'] == 40
    assert all(torch.equal(result[k], frozen[k]) for k in result)
    assert result['features'].shape == (22,)


def test_public_forward_rejects_private_payload():
    model, public = example(); public['labels'] = dict(target_xy=torch.zeros(2))
    with pytest.raises(ValueError):
        api().public_forward(model, public)


def test_cpu_diagnostic_does_not_update_any_model_parameter():
    model, public = example(); before = deepcopy(model.state_dict())
    result = api().public_forward(model, public)
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    assert not result['features'].requires_grad


def test_no_available_candidate_keeps_original_action_exactly():
    model, public = example(); public['network']['pair_valid'].zero_()
    result = api().public_forward(model, public)
    assert result['available'].item() is False
    assert torch.equal(result['before_xy'], result['after_xy'])
    assert api().private_outcomes(result, torch.tensor([10.,10.]))['outcome'] == 0


def test_nan_target_rejected_only_in_private_outcome_stage():
    model, public = example(); result = api().public_forward(model, public)
    with pytest.raises(ValueError):
        api().private_outcomes(result, torch.tensor([float('nan'),0.]))
