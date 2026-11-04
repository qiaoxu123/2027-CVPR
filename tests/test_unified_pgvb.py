"""CPU smoke tests for a single PGVB visual branch."""
from types import SimpleNamespace
import pytest
import torch
from graphnav.a3_runtime import A3TrajectoryPolicy
from graphnav.unified_pgvb import UnifiedPGVB
from graphnav.unified_pgvb_runtime import UnifiedPGVBNamedLandmarkPolicy


def _input():
    return dict(image_features=torch.randn(1, 4, 3, 1024),
                name_text=torch.randn(1, 1024), type_text=torch.randn(1, 1024),
                polygon_tokens=torch.randn(1, 16, 4),
                valid=torch.ones(1, 4, dtype=torch.bool))


def test_pgvb_head_is_same_network():
    model = UnifiedPGVB().eval()
    batch = _input()
    with torch.no_grad():
        actual = model(**batch)
        expected = model.pgvb(**batch)
    torch.testing.assert_close(actual["landmark_scores"], expected, atol=0, rtol=0)
    assert actual["target_visual_scores"] is None


def test_relation_and_target_heads_refuse_untrained_use():
    model = UnifiedPGVB().eval()
    with pytest.raises(RuntimeError, match="not trained"):
        model(**_input(), target_text=torch.randn(1, 1024))
    with pytest.raises(RuntimeError, match="not calibrated"):
        model(**_input(), anchor_world=torch.ones(1, 28, 28),
              direction_id=torch.tensor([0]), span=torch.ones(1, 2) * 300)
    model.direction_ready = True
    with torch.no_grad():
        prob = model(**_input(), anchor_world=torch.ones(1, 28, 28),
                     direction_id=torch.tensor([0]), span=torch.ones(1, 2) * 300)
    assert prob["direction_probability"].shape == (1, 28, 28)


def test_unified_policy_does_not_call_legacy_visual_model(monkeypatch):
    obj = object.__new__(UnifiedPGVBNamedLandmarkPolicy)
    obj.record = {"instruction": "Go to named landmark"}
    obj._initialize_geometry = lambda: None
    obj._timed_scores = lambda observation: (None, "no_local_anchor")
    obj.base = object()
    obj.audit_callback = None
    obj.pgvb_calls = 0
    obj.pgvb_seconds = 0.
    obj.unified = SimpleNamespace(direction_ready=False)
    monkeypatch.setattr(A3TrajectoryPolicy, "predict_field",
                        lambda *args: SimpleNamespace(diagnostics={}))
    result = obj.predict(SimpleNamespace(current_observation={}), obj.record["instruction"])
    assert result.diagnostics["old_stage1_rgb_executed"] is False
    assert result.diagnostics["siglip2_rgb_forward"] is False
