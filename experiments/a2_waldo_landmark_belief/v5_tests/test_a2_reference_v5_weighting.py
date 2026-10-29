import math

import pytest

from experiments.a2_waldo_landmark_belief.reference_v5_weighting import (
    target_balanced_weights, weighting_manifest,
)


def entries():
    rows = []
    groups = [
        ("revision_v1_A0_closedloop", "m1", 1, 3),
        ("revision_v1_A0_closedloop", "m1", 2, 1),
        ("revision_v2_A0_closedloop", "m2", 3, 2),
        ("teacher_snapshot", "m3", 4, 2),
        ("teacher_snapshot", "m3", 5, 1),
    ]
    for source_kind, map_name, object_id, count in groups:
        rows.extend(dict(role="train", source_kind=source_kind, map_name=map_name,
                         object_id=object_id) for _ in range(count))
    return rows


def test_each_target_group_gets_equal_total_weight_within_its_source_family():
    rows = entries()
    weights = target_balanced_weights(rows)

    group_totals = {}
    for row, weight in zip(rows, weights):
        key = row["source_kind"], row["map_name"], row["object_id"]
        group_totals[key] = group_totals.get(key, 0.0) + weight

    real_totals = [total for (source, _, _), total in group_totals.items()
                   if source != "teacher_snapshot"]
    teacher_totals = [total for (source, _, _), total in group_totals.items()
                      if source == "teacher_snapshot"]
    assert all(math.isclose(value, 15.6923076923, rel_tol=1e-8) for value in real_totals)
    assert all(math.isclose(value, 1.9615384615, rel_tol=1e-8) for value in teacher_totals)


def test_weights_preserve_v4_mean_row_weight_and_input_order():
    rows = entries()
    weights = target_balanced_weights(rows)
    assert len(weights) == len(rows)
    assert math.isclose(sum(weights), 51.0, rel_tol=1e-12)
    assert all(weight > 0 for weight in weights)
    assert weights[0] == weights[1] == weights[2]


@pytest.mark.parametrize("bad_row", [
    dict(role="dev", source_kind="teacher_snapshot", map_name="m", object_id=1),
    dict(role="train", source_kind="calibration_fit", map_name="m", object_id=1),
    dict(role="train", source_kind=[], map_name="m", object_id=1),
    dict(role="train", source_kind="teacher_snapshot", map_name="", object_id=1),
    dict(role="train", source_kind="teacher_snapshot", map_name="m", object_id=True),
    dict(role="train", source_kind="teacher_snapshot", map_name="m", object_id="1"),
])
def test_invalid_role_source_or_target_identity_is_rejected(bad_row):
    with pytest.raises(ValueError):
        target_balanced_weights([bad_row])


def test_empty_inventory_is_rejected():
    with pytest.raises(ValueError):
        target_balanced_weights([])


@pytest.mark.parametrize("kwargs", [
    {"real_weight": float("nan")},
    {"teacher_weight": float("inf")},
    {"real_weight": True},
])
def test_nonfinite_or_boolean_source_weights_are_rejected(kwargs):
    with pytest.raises(ValueError):
        target_balanced_weights(entries(), **kwargs)


def test_weighting_manifest_binds_population_and_exact_weight_bytes():
    rows = entries()
    weights = target_balanced_weights(rows)
    manifest = weighting_manifest(rows, weights)

    assert manifest["rows_by_source"] == {
        "revision_v1_A0_closedloop": 4,
        "revision_v2_A0_closedloop": 2,
        "teacher_snapshot": 3,
    }
    assert manifest["groups_by_source"] == {
        "revision_v1_A0_closedloop": 2,
        "revision_v2_A0_closedloop": 1,
        "teacher_snapshot": 2,
    }
    assert manifest["mean_weight"] == pytest.approx(51 / 9)
    assert len(manifest["weights_sha256"]) == 64


def test_weighting_manifest_rejects_misaligned_weight_population():
    with pytest.raises(ValueError):
        weighting_manifest(entries(), [1.0])
