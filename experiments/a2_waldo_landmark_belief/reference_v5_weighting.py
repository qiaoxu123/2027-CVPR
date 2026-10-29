"""Target-group loss weights for the isolated V5 training iteration."""
from collections import Counter
import hashlib
import json
import math
from numbers import Integral


REAL_SOURCES = frozenset({
    "revision_v1_A0_closedloop",
    "revision_v2_A0_closedloop",
})
TEACHER_SOURCE = "teacher_snapshot"


def target_balanced_weights(entries, *, real_weight=8.0, teacher_weight=1.0):
    """Return row weights with equal mass per source/map/target group.

    The shared scale preserves the mean of the V4 per-row source weights.
    Entries must be TRAIN manifest records; rows stay in their original order.
    """
    rows = list(entries)
    if not rows:
        raise ValueError("A nonempty TRAIN inventory is required")
    if (not isinstance(real_weight, (int, float)) or isinstance(real_weight, bool)
            or not isinstance(teacher_weight, (int, float)) or isinstance(teacher_weight, bool)
            or not math.isfinite(real_weight) or not math.isfinite(teacher_weight)
            or real_weight <= 0 or teacher_weight <= 0):
        raise ValueError("Source weights must be positive numbers")

    groups = []
    base_weights = []
    counts = Counter()
    for row in rows:
        if not isinstance(row, dict) or row.get("role") != "train":
            raise ValueError("Only TRAIN manifest rows may be weighted")
        source = row.get("source_kind")
        if not isinstance(source, str):
            raise ValueError("Unknown training source family")
        if source == TEACHER_SOURCE:
            base = float(teacher_weight)
        elif source in REAL_SOURCES:
            base = float(real_weight)
        else:
            raise ValueError("Unknown training source family")
        map_name, object_id = row.get("map_name"), row.get("object_id")
        if (not isinstance(map_name, str) or not map_name.strip() or "\0" in map_name
                or not isinstance(object_id, Integral) or isinstance(object_id, bool)):
            raise ValueError("Invalid map/target group identity")
        group = (source, map_name, int(object_id))
        groups.append(group)
        base_weights.append(base)
        counts[group] += 1

    v4_total = sum(base_weights)
    balanced_total = sum(base_weights[index] / counts[group]
                          for index, group in enumerate(groups))
    if balanced_total <= 0:
        raise ValueError("Invalid balanced source weight total")
    scale = v4_total / balanced_total
    return [base_weights[index] / counts[group] * scale
            for index, group in enumerate(groups)]


def weighting_manifest(entries, weights):
    """Describe the exact TRAIN population and normalized weights for provenance."""
    rows, values = list(entries), [float(value) for value in weights]
    if len(rows) != len(values) or not rows:
        raise ValueError("Weight manifest population mismatch")
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Weight manifest contains invalid values")
    row_counts, group_members, mass = Counter(), {}, Counter()
    for row, value in zip(rows, values):
        if not isinstance(row, dict) or row.get("role") != "train":
            raise ValueError("Weight manifest accepts TRAIN rows only")
        source = row.get("source_kind")
        if not isinstance(source, str):
            raise ValueError("Unknown source in weight manifest")
        if source not in REAL_SOURCES and source != TEACHER_SOURCE:
            raise ValueError("Unknown source in weight manifest")
        map_name, object_id = row.get("map_name"), row.get("object_id")
        if (not isinstance(map_name, str) or not map_name.strip() or "\0" in map_name
                or not isinstance(object_id, Integral) or isinstance(object_id, bool)):
            raise ValueError("Invalid group identity in weight manifest")
        key = (source, map_name, int(object_id))
        row_counts[source] += 1
        group_members.setdefault(key, []).append(value)
        mass[source] += value
    encoded = json.dumps(values, separators=(",", ":"), allow_nan=False).encode()
    return dict(
        rows_by_source=dict(sorted(row_counts.items())),
        groups_by_source=dict(sorted(Counter(source for source, _, _ in group_members).items())),
        weight_mass_by_source=dict(sorted(mass.items())),
        mean_weight=sum(values) / len(values),
        weights_sha256=hashlib.sha256(encoded).hexdigest(),
    )
