"""CPU tests: geometry is actually owned by UnifiedPGVB, not externally fused."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from graphnav.full_localization import GoalField
from graphnav.full_experiment import new_model
from graphnav.polygon_guided_visual import PolygonGuidedVisualBelief
from graphnav.pgvb_navigation_runtime import bounded_pgvb_field
from graphnav.unified_pgvb import UnifiedPGVB
from scripts.build_unified_pgvb_v2 import build_unified_checkpoint


class TinyGeometry(nn.Module):
    """Deterministic GoalField stand-in with an actual differentiable parameter."""
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(.1))
        self.calls = 0

    def forward(self, batch, *, variant, tokens, token_mask, node_emb):
        assert variant == "geometry"
        self.calls += 1
        xy = batch["grid_xy"]
        b, q, _ = xy.shape
        logits = (
            torch.linspace(0, 1, q, device=xy.device)[None].expand(b, -1)
            + self.bias
        )
        dense = xy[..., :1].expand(-1, -1, 128).contiguous()
        ids = logits.topk(20, dim=1).indices
        return GoalField(
            belief_logits=logits,
            candidate_ids=ids,
            candidate_xy=xy.gather(1, ids[..., None].expand(-1, -1, 2)),
            candidate_context=dense.gather(
                1, ids[..., None].expand(-1, -1, 128),
            ),
            dense_context=dense,
        )


def _inputs():
    v = (torch.arange(28) + .5) / 28
    ys, xs = torch.meshgrid(v, v, indexing="ij")
    grid = torch.stack((xs.flatten(), ys.flatten()), -1)[None]
    return dict(static_inputs={"grid_xy": grid},
                tokens=None, token_mask=None, node_emb=None)


def _observed():
    uv = torch.tensor([[
        [.15, .15], [.16, .15], [.50, .50], [.51, .50],
        [.80, .80], [.81, .80], [.10, .85], [.11, .85],
    ]], dtype=torch.float32)
    return dict(
        local_uv=uv,
        image_features=torch.randn(1, 8, 3, 1024),
        name_text=torch.randn(1, 1024),
        type_text=torch.randn(1, 1024),
        polygon_tokens=torch.randn(1, 16, 4),
        valid=torch.ones(1, 8, dtype=torch.bool),
    )


def test_unified_field_owns_geometry_and_preserves_controller_contract():
    geometry = TinyGeometry()
    model = UnifiedPGVB(geometry=geometry).eval().requires_grad_(False)
    base = model.forward_field(**_inputs())
    assert base["field"] is base["geometry_field"]
    assert geometry.calls == 1
    assert base["field"].dense_context.shape == (1, 784, 128)
    assert base["field"].candidate_context.shape == (1, 20, 128)
    # Frozen geometry is cached per episode only.
    model.forward_field(**_inputs())
    assert geometry.calls == 1
    model.reset_episode()
    model.forward_field(**_inputs())
    assert geometry.calls == 2


def test_integrated_visual_fusion_matches_legacy_without_external_a3():
    torch.manual_seed(20261104)
    model = UnifiedPGVB(geometry=TinyGeometry()).eval().requires_grad_(False)
    inputs = _inputs()
    observed = _observed()
    result = model.forward_field(**inputs, observed=observed)
    raw = model.pgvb(**{
        k: v for k, v in observed.items() if k != "local_uv"
    }).float()
    legacy, legacy_audit = bounded_pgvb_field(
        result["geometry_field"], inputs["static_inputs"]["grid_xy"],
        observed["local_uv"], raw, observed["valid"],
    )
    torch.testing.assert_close(
        result["field"].belief_logits, legacy.belief_logits, atol=0, rtol=0,
    )
    torch.testing.assert_close(
        result["field"].candidate_context, legacy.candidate_context,
        atol=0, rtol=0,
    )
    assert result["audit"]["active"] == legacy_audit["active"]
    assert model.geometry.calls == 1


def test_trainable_geometry_does_not_cache_and_gets_gradient():
    model = UnifiedPGVB(geometry=TinyGeometry()).train()
    model.reset_episode()
    a = model.forward_field(**_inputs())["field"].belief_logits.square().sum()
    a.backward()
    assert model.geometry.bias.grad is not None
    assert torch.isfinite(model.geometry.bias.grad)
    model.forward_field(**_inputs())
    assert model.geometry.calls == 2


def test_visual_projection_receives_gradients_in_single_network():
    model = UnifiedPGVB(geometry=TinyGeometry()).train()
    observed = _observed()
    scores = model.pgvb(**{
        k: v for k, v in observed.items() if k != "local_uv"
    })
    scores.square().sum().backward()
    assert model.pgvb.image[0].weight.grad is not None


def test_v2_builder_preserves_exact_stage1_geometry_weights(tmp_path):
    args = SimpleNamespace(
        hidden=128, heads=4, max_relations=8, vision_dim=1024,
        top_k=20, rule_weight=1., grid_size=28,
    )
    geometry = new_model(args, torch.device("cpu"))
    stage1_path = tmp_path / "stage1.pt"
    pgvb_path = tmp_path / "pgvb.pt"
    output = tmp_path / "unified.pt"
    torch.save(dict(
        protocol=dict(name="graphnav-v1-stage1"),
        a3_args=vars(args),
        model={"a3." + key: value for key, value in geometry.state_dict().items()},
    ), stage1_path)
    torch.save(dict(
        config=dict(no_geometry_belief=True),
        model=PolygonGuidedVisualBelief().state_dict(),
    ), pgvb_path)
    report = build_unified_checkpoint(stage1_path, pgvb_path, output)
    assert report["geometry_parameters"] > 0
    payload = torch.load(output, map_location="cpu", weights_only=True)
    assert payload["format"] == "unified-pgvb-v2"
    assert payload["config"]["geometry_integrated"]
    assert not payload["config"]["jointly_trained"]
    keys = payload["model"]
    assert "geometry.token_head.1.weight" in keys
    assert "geometry.vision_proj.0.weight" not in keys
    torch.testing.assert_close(
        keys["geometry.token_head.1.weight"],
        geometry.state_dict()["token_head.1.weight"],
        rtol=0, atol=0,
    )
    with pytest.raises(FileExistsError):
        build_unified_checkpoint(stage1_path, pgvb_path, output)
