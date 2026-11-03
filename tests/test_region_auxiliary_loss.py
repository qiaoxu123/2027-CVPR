import inspect
import torch

from graphnav.region_auxiliary_loss import RegionAuxiliaryHead, region_auxiliary_loss
from graphnav.visual_chain_position import WorldLikelihoodPositionHead


def test_auxiliary_reaches_shared_visual_projection_only():
    model = WorldLikelihoodPositionHead(dimension=8, hidden=4)
    head = RegionAuxiliaryHead(hidden=4)
    features = torch.randn(1, 3, 3, 8)
    logits = head(image_features=features, projection=model.image)
    loss, count = region_auxiliary_loss(logits, torch.tensor([[0, 1, -1]]),
        torch.ones(1, 3, dtype=torch.bool), torch.ones(3))
    loss.backward()
    assert count == 2 and model.image.weight.grad.norm() > 0
    assert model.text.weight.grad is None and model.geometry.weight.grad is None
    assert head.classifier.weight.grad.norm() > 0


def test_unknown_or_invalid_points_do_not_become_background():
    logits = torch.randn(1, 3, 3, requires_grad=True)
    label = torch.tensor([[0, 1, -1]])
    loss, count = region_auxiliary_loss(logits, label, torch.tensor([[True, False, True]]), torch.ones(3))
    loss.backward()
    assert count == 1
    assert not logits.grad[0, 1:].any()
    empty, count = region_auxiliary_loss(logits, torch.full_like(label, -1), torch.ones_like(label, dtype=torch.bool), torch.ones(3))
    assert count == 0 and torch.isfinite(empty) and empty == 0


def test_inference_architecture_and_parameters_remain_compatible():
    model = WorldLikelihoodPositionHead(dimension=8, hidden=4)
    auxiliary = RegionAuxiliaryHead(hidden=4)
    exported = model.state_dict()
    restored = WorldLikelihoodPositionHead(dimension=8, hidden=4)
    restored.load_state_dict(exported, strict=True)
    assert 'region_labels' not in inspect.signature(model.forward).parameters
    assert set(inspect.signature(auxiliary.forward).parameters) == {'image_features', 'projection'}
    assert not any('auxiliary' in key for key in exported)
