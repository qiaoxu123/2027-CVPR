import numpy as np
from graphnav.training_supervision_audit import position_region_counts, car_core_position_counts


def test_coarse_world_positive_can_disagree_with_region_proxy():
    counts = position_region_counts(np.array([0., 10., 20., 21.]), np.ones(4, bool),
                                   np.array([0, 1, -1, 2]), 'Building')
    assert counts['positive_points'] == 3
    assert counts['same_region'] == 1 and counts['other_region'] == 1 and counts['unknown_region'] == 1


def test_car_inside_parking_is_not_a_semantic_contradiction():
    counts = position_region_counts(np.array([1.]), np.ones(1, bool), np.array([1]), 'Car')
    assert counts['applicable'] is False
    assert 'other_region' not in counts


def test_known_other_car_cores_are_counted_separately_from_unknown():
    counts = car_core_position_counts(np.array([1., 5., 21., 3.]), np.ones(4, bool), np.array([1, 0, 0, -1]))
    assert counts['target_core_points'] == 1 and counts['other_car_core_points'] == 1
    assert counts['unknown_points'] == 1 and counts['position_positive_points'] == 3
