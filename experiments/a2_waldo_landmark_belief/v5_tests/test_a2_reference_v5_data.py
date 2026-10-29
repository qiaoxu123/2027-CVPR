import torch
import pytest

from experiments.a2_waldo_landmark_belief import reference_v5_data


class TinyReferenceViews:
    def __init__(self, role, *, directory=None):
        self.entries = [
            dict(role=role, source_kind="revision_v1_A0_closedloop", map_name="m", object_id=1),
            dict(role=role, source_kind="revision_v1_A0_closedloop", map_name="m", object_id=1),
            dict(role=role, source_kind="teacher_snapshot", map_name="n", object_id=2),
        ]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        base_weight = 8.0 if self.entries[index]["source_kind"] != "teacher_snapshot" else 1.0
        return {"labels": {"sample_weight": torch.tensor(base_weight)}}


def test_train_wrapper_replaces_only_sample_weight_and_preserves_row_order(monkeypatch):
    monkeypatch.setattr(reference_v5_data, "ReferenceOverlayViews", TinyReferenceViews)

    dataset = reference_v5_data.TargetBalancedReferenceViews("train")
    first, second, teacher = dataset[0], dataset[1], dataset[2]

    assert first["labels"]["sample_weight"].item() == second["labels"]["sample_weight"].item()
    assert first["labels"]["sample_weight"].item() == pytest.approx(7.555555555555555)
    assert teacher["labels"]["sample_weight"].item() == pytest.approx(1.8888888888888888)
    assert len(dataset) == 3


def test_dev_wrapper_does_not_apply_train_group_balancing(monkeypatch):
    monkeypatch.setattr(reference_v5_data, "ReferenceOverlayViews", TinyReferenceViews)

    dataset = reference_v5_data.TargetBalancedReferenceViews("dev")

    assert dataset[0]["labels"]["sample_weight"].item() == 8.0
    assert dataset[2]["labels"]["sample_weight"].item() == 1.0
