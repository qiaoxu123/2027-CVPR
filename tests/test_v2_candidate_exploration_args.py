import pytest

from scripts.evaluate_v2_expanded_public_navigation import validate_exploration_args


def test_candidate_exploration_is_bounded_and_geometry_selected_only():
    validate_exploration_args(0, "geometry-top1")
    validate_exploration_args(4, "geometry-top1")

    with pytest.raises(ValueError, match="between 0 and 8"):
        validate_exploration_args(9, "geometry-top1")
    with pytest.raises(ValueError, match="only defined for geometry-top1"):
        validate_exploration_args(4, "ranked-once")
