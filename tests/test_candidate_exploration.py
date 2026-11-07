import numpy as np

from graphnav.candidate_exploration import spatial_candidate_slots


def test_candidate_search_keeps_score_order_and_removes_nearby_cells():
    candidates = np.array([[0.0, 0.0], [5.0, 0.0], [21.0, 0.0], [50.0, 0.0]])

    assert spatial_candidate_slots(candidates, max_candidates=4, min_distance_m=20.0) == [0, 2, 3]


def test_candidate_search_handles_small_or_invalid_inputs():
    assert spatial_candidate_slots(np.array([[0.0, 0.0]]), max_candidates=0, min_distance_m=20.0) == []
    assert spatial_candidate_slots(np.empty((0, 2)), max_candidates=4, min_distance_m=20.0) == []
    assert spatial_candidate_slots(np.array([[0.0, 0.0]]), max_candidates=4, min_distance_m=20.0) == [0]
