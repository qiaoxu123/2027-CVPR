import copy

import pytest

from experiments.a2_waldo_landmark_belief.reference_v4_protocol import CONFIG as V4_CONFIG
from experiments.a2_waldo_landmark_belief.reference_v5_protocol import (
    CONFIG, validate_training_config,
)


def test_v5_changes_only_source_weighting_and_preserves_v4_schedule():
    validate_training_config()
    for key, value in V4_CONFIG.items():
        if key != "version":
            assert CONFIG[key] == value
    assert CONFIG["version"] == "a2_relation_v5_target_group_balanced"
    assert CONFIG["target_weighting"] == "equal group mass within source; real:teacher=8:1"


def test_v5_rejects_metadata_that_changes_the_frozen_v4_training_controls(monkeypatch):
    validate_training_config()
    monkeypatch.setitem(CONFIG, "rank_epochs", 19)
    with pytest.raises(ValueError):
        validate_training_config()


def test_v5_rejects_changed_target_weighting_description(monkeypatch):
    validate_training_config()
    monkeypatch.setitem(CONFIG, "target_weighting", "row-balanced")
    with pytest.raises(ValueError):
        validate_training_config()
