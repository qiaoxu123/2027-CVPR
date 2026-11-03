import math
import torch
from scripts.audit_spatial_prior_feedback import uniform_prior_feature


def test_intervention_only_changes_last_spatial_feature_without_mutation():
    x = torch.randn(2, 144, 71)
    old = x.clone()
    changed, = uniform_prior_feature(None, (x,))
    assert torch.equal(x, old)
    assert torch.equal(changed[..., :-1], old[..., :-1])
    assert torch.all(changed[..., -1] == -math.log(784))
