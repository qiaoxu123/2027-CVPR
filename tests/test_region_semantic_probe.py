import inspect
import numpy as np
import torch
from graphnav.region_semantic_probe import RegionSemanticProbe, classification_metrics, shuffle_observed_features


def test_visual_probe_gradient_and_input_whitelist():
    model = RegionSemanticProbe(dimension=8)
    x = torch.randn(4, 3, 8)
    loss = torch.nn.functional.cross_entropy(model(image_features=x), torch.tensor([0, 1, 2, 0]))
    loss.backward()
    assert model.classifier.weight.grad.norm() > 0
    assert set(inspect.signature(model.forward).parameters) == {'image_features'}


def test_shuffle_uses_public_validity_not_private_labels():
    feature = torch.arange(24).view(3, 1, 8)
    valid = torch.tensor([True, False, True])
    result = shuffle_observed_features(feature, valid)
    assert torch.equal(result[0], feature[2]) and torch.equal(result[1], feature[1])
    assert set(inspect.signature(shuffle_observed_features).parameters) == {'features', 'valid'}


def test_metrics_report_class_balanced_performance():
    result = classification_metrics(np.array([0, 0, 0, 1, 2]), np.array([0, 0, 0, 0, 2]))
    assert result['accuracy'] == .8
    assert result['balanced_accuracy'] == 2/3
    assert result['confusion'] == [[3, 0, 0], [1, 0, 0], [0, 0, 1]]
