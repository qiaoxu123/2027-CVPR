"""GT-free unified language adapter and single PGVB routing regression tests."""
from types import SimpleNamespace

import pytest
import torch

from graphnav.unified_instruction import (
    parse_unified_instruction, compile_structured_instruction,
)
from graphnav.unified_pgvb import UnifiedPGVB
from graphnav.unified_pgvb_runtime import UnifiedPGVBNamedLandmarkPolicy


def _objects():
    return {
        "a": dict(
            name="The Food Warehouse", object_type="Building",
            contour=[[0, 0], [8, 0], [8, 8], [0, 8]],
        ),
        "b": dict(
            name="Victoria Street", object_type="TrafficRoad",
            contour=[[0, 0], [30, 0], [30, 3], [0, 3]],
        ),
        "c": dict(
            name="Victoria Street", object_type="Footpath",
            contour=[[0, 4], [30, 4], [30, 5], [0, 5]],
        ),
    }


def _image_input():
    torch.manual_seed(42)
    return dict(
        image_features=torch.randn(1, 5, 3, 1024),
        name_text=torch.randn(1, 1024),
        type_text=torch.randn(1, 1024),
        polygon_tokens=torch.randn(1, 16, 4),
        valid=torch.ones(1, 5, dtype=torch.bool),
    )


def test_named_goal_has_explicit_intent_and_no_gt_or_relation():
    instruction = "Navigate to The Food Warehouse."
    spec = parse_unified_instruction(instruction, "toy", _objects())
    assert spec["approved_for_inference"] is True
    assert spec["intent"] == "navigate_to_landmark"
    assert spec["target"]["same_as_landmark"] is True
    assert spec["landmark"]["name"] == "The Food Warehouse"
    assert spec["landmark"]["public_ids"] == ["a"]
    assert spec["relations"] == []
    assert spec["target_GT_used"] is False
    assert instruction[spec["source_span"][0]:spec["source_span"][1]].startswith("The Food Warehouse")


def test_same_name_road_parts_are_preserved_without_gt_selection():
    spec = parse_unified_instruction("Go to Victoria Street.", "toy", _objects())
    assert spec["approved_for_inference"] is True
    assert spec["landmark"]["type"] == "Road"
    assert set(spec["landmark"]["public_ids"]) == {"b", "c"}


def test_unknown_or_mixed_name_never_navigates_as_if_confirmed():
    assert not parse_unified_instruction(
        "Navigate to Unknown Place.", "toy", _objects()
    )["approved_for_inference"]
    objects = _objects()
    objects["d"] = dict(
        name="Victoria Street", object_type="Building",
        contour=[[0, 10], [4, 10], [4, 14], [0, 14]],
    )
    s = parse_unified_instruction("Go to Victoria Street.", "toy", objects)
    assert s["intent"] == "navigate_to_landmark"
    assert s["approved_for_inference"] is True
    assert s["reason"] == "explicit_named_goal_geometry_only"
    assert s["landmark"]["visual_supported"] is False
    assert len(s["landmark"]["public_ids"]) == 3
    assert "mixed_or_unsupported_public_landmark_type" in s["ambiguity"]


def test_complex_target_is_not_silently_the_public_landmark_goal():
    spec = parse_unified_instruction(
        "Find the white building north of The Food Warehouse.",
        "toy", _objects(),
    )
    assert not spec["approved_for_inference"]
    assert spec["intent"] == "unresolved"
    assert spec["reason"] == "complex_instruction_needs_parser"


def test_rich_parser_retains_direction_scope_and_never_enables_target_head():
    raw = dict(
        original_text="Find the white building north of The Food Warehouse.",
        map_name="toy", target_GT_used=False,
        target=dict(category="Building", attributes=[
            dict(kind="color", value="white", span=[9, 14])
        ]),
        landmarks=[dict(start=33, end=51, public_names=["The Food Warehouse"],
                        public_ids=["a"], types=["Building"])],
        relations=[dict(
            id="r0", predicate="north of", source_span=[9, 51],
            relation_kind="spatial_proposal",
            reference_frame=dict(frame="world_cardinal"), ambiguity=[],
        )],
        direct_relations=[dict(
            hops=1, landmark_names=["The Food Warehouse"],
            relations=[dict(relation_id="r0", traversal="forward")],
            ambiguity=[],
        )],
        indirect_relations=[], ambiguity=[], parser_status="automatic_unverified",
    )
    spec = compile_structured_instruction(raw)
    assert spec["intent"] == "direct_target"
    assert spec["target"]["attributes"][0]["value"] == "white"
    assert spec["relations"][0]["chain"][0]["predicate"] == "north of"
    assert spec["relations"][0]["chain"][0]["traversal"] == "forward"
    assert spec["landmarks"][0]["ids"] == ["a"]
    assert "geographic_north_not_independently_verified" in spec["ambiguity"]
    assert not spec["approved_for_inference"]
    with pytest.raises(ValueError, match="GT-free"):
        compile_structured_instruction(dict(raw, target_GT_used=True))


def test_unified_network_accepts_valid_named_intent_only():
    model = UnifiedPGVB().eval()
    spec = parse_unified_instruction("Navigate to The Food Warehouse.", "toy", _objects())
    with torch.no_grad():
        baseline = model(**_image_input())
        routed = model.forward_structured(spec, **_image_input())
    torch.testing.assert_close(
        routed["landmark_scores"], baseline["landmark_scores"], atol=0, rtol=0,
    )
    direct = dict(spec, intent="direct_target", approved_for_inference=False)
    with pytest.raises(ValueError, match="Only validated"):
        model.forward_structured(direct, **_image_input())


def test_runtime_refuses_target_reference_confusion_before_geometry(monkeypatch):
    policy = object.__new__(UnifiedPGVBNamedLandmarkPolicy)
    catalog = SimpleNamespace(objects={"toy": _objects()})
    target = SimpleNamespace(
        instruction="Find the white building north of The Food Warehouse.",
        map_name="toy", reference_names=("The Food Warehouse",),
    )
    with pytest.raises(ValueError, match="refuses non-landmark"):
        policy.set_episode(target, None, catalog)
    wrong = SimpleNamespace(
        instruction="Navigate to The Food Warehouse.",
        map_name="toy", reference_names=("Victoria Street",),
    )
    with pytest.raises(ValueError, match="differ"):
        policy.set_episode(wrong, None, catalog)


def test_runtime_has_no_legacy_stage1_or_parallel_pgvb_inheritance():
    from graphnav.stage1_runtime import GraphNavStage1Policy
    from graphnav.pgvb_navigation_runtime import PGVBNamedLandmarkPolicy
    assert not issubclass(UnifiedPGVBNamedLandmarkPolicy, GraphNavStage1Policy)
    assert not issubclass(UnifiedPGVBNamedLandmarkPolicy, PGVBNamedLandmarkPolicy)
    assert "_pgvb_forward" not in UnifiedPGVBNamedLandmarkPolicy.__dict__
    assert "_observed_inputs" in UnifiedPGVBNamedLandmarkPolicy.__dict__

def test_ground_landmark_uses_public_geometry_only():
    objects = _objects()
    objects["ground"] = dict(
        name="Station Square", object_type="Ground",
        contour=[[40, 40], [70, 40], [70, 70], [40, 70]],
    )
    spec = parse_unified_instruction("Navigate to Station Square.", "toy", objects)
    assert spec["approved_for_inference"]
    assert spec["landmark"]["type"] == "Other"
    assert not spec["landmark"]["visual_supported"]
    assert spec["target_GT_used"] is False

def test_mixed_type_named_goal_runtime_uses_geometry_only(monkeypatch):
    from graphnav import unified_pgvb_runtime as runtime
    objects = _objects()
    objects["d"] = dict(
        name="Victoria Street", object_type="Building",
        contour=[[0, 10], [4, 10], [4, 14], [0, 14]],
    )
    monkeypatch.setattr(
        runtime, "public_static_record",
        lambda *args, **kwargs: dict(
            map_name="toy",
            instruction="Navigate to Victoria Street.",
            world_bounds=[0., 0., 100., 100.],
        ),
    )
    policy = object.__new__(runtime.UnifiedPGVBNamedLandmarkPolicy)
    policy.args = SimpleNamespace(max_landmarks=16)
    policy.mode = "tracked-stable"
    policy.native = SimpleNamespace()
    policy.device = torch.device("cpu")
    policy.unified = SimpleNamespace(reset_episode=lambda: None)
    # Production initializes both trackers; bypassing __init__ in this unit test
    # must explicitly provide their episode-reset interfaces.
    policy.acquisition = SimpleNamespace(reset=lambda: None)
    policy.stable_goal = SimpleNamespace(reset=lambda: None)
    episode = SimpleNamespace(
        instruction="Navigate to Victoria Street.",
        map_name="toy",
        reference_names=("Victoria Street",),
    )
    policy.set_episode(
        episode, None, SimpleNamespace(objects={"toy": objects}),
    )
    assert policy.instruction_spec["approved_for_inference"]
    assert policy._landmark is None
    assert policy.inputs is None
    evidence, reason = policy._observed_inputs({})
    assert evidence is None
    assert reason == "unsupported_or_unresolved_name"
