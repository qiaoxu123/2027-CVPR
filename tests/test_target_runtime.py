"""Native current-frame coordinate projection and supervised adapter smoke."""
import numpy as np
import pytest
import torch

from graphnav.target_runtime import native_patch_world_xy
from graphnav.evidence_grounding import TargetEvidenceRefiner
from graphnav.train_target_evidence import evidence_loss, load_rows


def test_observed_native_patch_centers_respect_support_and_world_bounds():
    image = np.ones((384, 384, 3), dtype=np.uint8) * 180
    valid = np.ones((384, 384), dtype=bool)
    valid[:192] = False
    obs = dict(
        bgr=image, depth_valid=valid,
        footprint_world=np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32),
    )
    xy, support = native_patch_world_xy(obs, 4, [0, 0, 100, 100], device="cpu")
    assert xy.shape == (1, 4, 2)
    assert support.shape == (1, 4)
    assert support.tolist() == [[False, False, True, True]]
    assert torch.allclose(xy[0, 0], torch.tensor([.25, .25]), atol=.01)


def test_training_labels_cannot_change_adapter_forward():
    torch.manual_seed(11)
    refiner = TargetEvidenceRefiner(hidden=8, text_dim=12, vision_dim=6, relation_dim=5)
    row = dict(
        base_logits=torch.randn(1, 16),
        dense_context=torch.randn(1, 16, 8),
        grid_xy=torch.rand(1, 16, 2),
        relation_fields=torch.randn(1, 16, 5),
        text_tokens=torch.randn(1, 4, 12),
        text_mask=torch.ones(1, 4, dtype=torch.bool),
        patch_features=torch.randn(1, 5, 6),
        patch_xy=torch.rand(1, 5, 2),
        patch_valid=torch.ones(1, 5, dtype=torch.bool),
        target_distribution=torch.nn.functional.one_hot(torch.tensor([2]), 16).float(),
    )
    a, _ = evidence_loss(refiner, row)
    modified = dict(row, target_distribution=torch.nn.functional.one_hot(torch.tensor([9]), 16).float())
    b, _ = evidence_loss(refiner, modified)
    assert a != b
    a.backward()
    assert refiner.relation_gate.grad is not None
    assert refiner.visual_gate.grad is not None


def test_nontraining_rows_are_rejected(tmp_path):
    path = tmp_path / "rows.pt"
    torch.save(dict(split="val_unseen", causal_observations=True, rows=[{}]), path)
    with pytest.raises(ValueError, match="train_seen"):
        load_rows(path)
    torch.save(dict(split="train_seen", causal_observations=False, rows=[{}]), path)
    with pytest.raises(ValueError, match="causal"):
        load_rows(path)
