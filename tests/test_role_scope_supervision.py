"""CPU checks for public-only role weak supervision and exact old-model behavior."""
import copy
import pytest
import torch

from graphnav.relation_scope_supervision import (
    public_role_targets, role_set_loss, role_type,
)


def public_scene():
    return dict(node_ids=["anchor", "parkingA", "parkingB", "carA", "carB"],
                valid=torch.tensor([True]*5), types=torch.tensor([1, 5, 5, 7, 7]))


def query():
    return dict(anchor_indices=torch.tensor([0, 0]),
                anchor_valid=torch.tensor([True, False]))


def indirect():
    car = dict(node=0, text="the blue car", kind="ordinary_object", public_ids=[])
    parking = dict(node=2, text="the parking lot", kind="ordinary_object", public_ids=[])
    anchor = dict(node=5, text="One Stop", kind="named_landmark", public_ids=["anchor"])
    return dict(paths=[dict(hops=2, semantic_confident=True, public_ids=["anchor"],
        nodes=[car, parking, anchor], steps=[
            dict(subject=car, object=parking, relation="in", orientation=1),
            dict(subject=parking, object=anchor, relation="near", orientation=1)])])


def test_two_hop_role_order_and_weak_type_set():
    res, stats = public_role_targets(indirect(), public_scene(), query())
    assert [(v["step"], v["kind"], v["candidates"]) for v in res] == [
        (0, "typed_intermediate", [1, 2]), (1, "typed_target", [3, 4])]
    assert stats["typed_intermediate"] == 1


def test_named_intermediate_uses_public_ids_not_nearest_gt():
    parsed = indirect()
    entity = parsed["paths"][0]["nodes"][1]
    entity["public_ids"] = ["parkingB"]
    entity["kind"] = "named_landmark"
    res, _ = public_role_targets(parsed, public_scene(), query())
    assert res[0]["kind"] == "named_public"
    assert res[0]["candidates"] == [2]


def test_scope_mismatch_and_ambiguous_paths_abstain():
    parsed = indirect()
    parsed["paths"][0]["steps"][1]["orientation"] = -1
    res, stats = public_role_targets(parsed, public_scene(), query())
    assert len(res) == 1 and stats["invalid_or_weak_scope"] == 1
    parsed = indirect()
    alternate = copy.deepcopy(parsed["paths"][0])
    alternate["nodes"][1]["node"] = 3
    parsed["paths"].append(alternate)
    res, stats = public_role_targets(parsed, public_scene(), query())
    assert not res and stats["ambiguous_path_scope"] == 1


def test_gt_changes_cannot_change_public_role_labels():
    parsed = indirect()
    a, _ = public_role_targets(parsed, public_scene(), query())
    parsed["target_xy"] = [0.1, 0.9]
    parsed["target_id"] = "GT-secret"
    b, _ = public_role_targets(parsed, public_scene(), query())
    assert a == b


def test_marginal_set_loss_backprop_and_empty_fallback():
    logits = torch.randn(2, 1, 2, 5, requires_grad=True)
    probs = logits.softmax(-1)
    label = dict(anchor=0, step=0, candidates=[1, 2],
                 kind="typed_intermediate", weight=.25)
    loss, n = role_set_loss(probs, [[label], []])
    assert n == 1 and torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0
    zero, count = role_set_loss(probs, [[], []])
    assert zero.item() == 0 and count == 0
    with pytest.raises(ValueError):
        role_set_loss(probs, [[dict(anchor=0, step=0, candidates=[99], weight=.1)], []])


def test_type_mapping_is_not_target_id_or_world_angle():
    assert role_type("the parking lot") == 5
    assert role_type("the blue van") == 7
    assert role_type("in front of the church") == 1
    assert role_type("unknown object") is None


def test_per_hop_output_preserves_original_and_has_gradients():
    from graphnav.relation_bearing_network import RolePolygonBearing
    torch.manual_seed(42)
    b, r, n, steps, dim = 2, 2, 5, 3, 8
    x = dict(tokens=torch.randn(b, 7, dim),
             token_mask=torch.ones(b, 7, dtype=torch.bool),
             node_names=torch.randn(b, n, dim), node_shape=torch.randn(b, n, 39),
             node_types=torch.zeros(b, n, dtype=torch.long),
             node_valid=torch.ones(b, n, dtype=torch.bool),
             edges=torch.randn(b, n, n, 10),
             anchor_indices=torch.tensor([[0, 1], [0, 1]]),
             anchor_valid=torch.tensor([[True, True], [True, False]]),
             entity_steps=torch.randn(b, r, steps, dim),
             relation_steps=torch.randn(b, r, steps, dim),
             step_valid=torch.tensor([[[True, True, False], [True, False, False]],
                                      [[True, True, True], [False, False, False]]]),
             angle_kernel=torch.rand(b, r, n, 8).softmax(-1))
    model = RolePolygonBearing(text_dim=dim, hidden=16, shape_dim=39)
    old = model(**x)
    new = model(**x, emit_steps=True)
    assert new["step_object_probabilities"].shape == (b, r, steps, n)
    torch.testing.assert_close(old["probabilities"], new["probabilities"], atol=0, rtol=0)
    torch.testing.assert_close(old["object_probabilities"], new["object_probabilities"],
                               atol=0, rtol=0)
    targets = [[dict(anchor=0, step=0, candidates=[1, 2],
                     kind="typed_intermediate", weight=.25)], []]
    loss, count = role_set_loss(new["step_object_probabilities"], targets)
    assert count == 1 and torch.isfinite(loss)
    loss.backward()
    assert model.relation.weight.grad.abs().sum() > 0
    assert model.edge[0].weight.grad.abs().sum() > 0
