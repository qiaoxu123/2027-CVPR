import inspect
import torch

from graphnav.visual_chain_position import WorldLikelihoodPositionHead
from graphnav.world_relation_factor import WorldRelationFactor


def inputs():
    torch.manual_seed(7)
    return dict(prior_logits=torch.randn(2, 4), local_uv=torch.rand(2, 3, 2),
                image_features=torch.randn(2, 3, 3, 8), target_text=torch.randn(2, 8),
                context_text=torch.randn(2, 8), public_geometry=torch.randn(2, 3, 31),
                quality=torch.ones(2, 3, 6), local_valid=torch.ones(2, 3, dtype=torch.bool),
                world_hypotheses=torch.randn(2, 7, 2)*20,
                reference_text=torch.randn(2, 2, 8), relation_ids=torch.tensor([[6, 7], [6, 7]]),
                reference_valid=torch.ones(2, 2, dtype=torch.bool),
                reference_geometry=torch.zeros(2, 2, 3, 31),
                reference_map_types=torch.zeros(2, 2, 17),
                reference_support=torch.ones(2, 2, 3, dtype=torch.bool))


def model():
    return WorldRelationFactor(WorldLikelihoodPositionHead(dimension=8, hidden=8), dimension=8, hidden=8)


def test_no_reference_exact_fallback_and_no_observation_safe():
    m = model(); x = inputs(); x['reference_valid'].zero_()
    public = {k: x[k] for k in m.BASE_INPUTS}
    assert torch.equal(m(**x), m.base(**public))
    x['reference_valid'].fill_(True); x['local_valid'].zero_()
    assert torch.equal(m(**x), m.base(**{k: x[k] for k in m.BASE_INPUTS}))
    assert torch.isfinite(m(**x)[:, :4]).all()


def test_actual_relation_and_reference_matching_gradients_with_frozen_base():
    m = model(); x = inputs()
    output, details = m(**x, return_details=True)
    output[:, 0].sum().backward()
    assert all(p.grad is None for p in m.base.parameters())
    assert m.reference_image.weight.grad.norm() > 0
    assert m.pair_geometry[0].weight.grad.norm() > 0
    assert m.relation_embedding.weight.grad.norm() > 0
    assert details['reference_mass'].shape == (2, 2)


def test_hidden_relation_control_invariant_to_codes():
    m = model(); x = inputs()
    x['reference_map_types'][..., 6] = 1
    before = m(**x, relation_mode='hidden')
    x['relation_ids'].fill_(10); x['reference_map_types'].zero_()
    x['reference_map_types'][..., 10] = 1
    assert torch.equal(before, m(**x, relation_mode='hidden'))


def test_invalid_reference_content_has_no_effect():
    m = model(); x = inputs(); x['reference_support'][:, 1].zero_()
    before = m(**x)
    x['reference_text'][:, 1] *= -10
    assert torch.equal(before, m(**x))


def test_joint_reference_permutation_invariant():
    m = model(); x = inputs(); before = m(**x)
    for key in ('reference_text', 'relation_ids', 'reference_valid', 'reference_geometry',
                'reference_map_types', 'reference_support'):
        x[key] = x[key].flip(1)
    torch.testing.assert_close(before, m(**x))


def test_no_gt_forward_arguments_and_invalid_mode_rejected():
    assert not {'target_world', 'target_id', 'contour', 'instance_labels'} & set(inspect.signature(WorldRelationFactor.forward).parameters)
    m = model(); x = inputs()
    import pytest
    with pytest.raises(ValueError):
        m(**x, relation_mode='not-a-control')
