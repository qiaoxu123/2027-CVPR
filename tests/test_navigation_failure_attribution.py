import numpy as np

from scripts.analyze_v2_navigation_failure_attribution import (
    primary_failure,
    relation_proxy,
    spatial_nms,
    target_nms_rank,
)


def test_spatial_nms_suppresses_points_inside_20m_but_keeps_distinct_peaks():
    xy = np.asarray([[0, 0], [5, 0], [21, 0], [60, 0]], dtype=float)
    assert spatial_nms([0, 1, 2, 3], xy, radius=20, k=4) == [0, 2, 3]


def test_target_rank_uses_posthoc_target_and_returns_score_gap():
    xy = np.asarray([[0, 0], [30, 0], [60, 0], [90, 0]], dtype=float)
    logits = np.asarray([5, 4, 3, 2], dtype=float)
    out = target_nms_rank(logits, xy, [30, 0], [0, 1, 2, 3], radius=20, k=4)
    assert out["rank"] == 2
    assert out["nms_hits"]["1"] is False
    assert out["nms_hits"]["5"] is True
    assert out["score_gap_to_target20"] == 1.0


def test_primary_failure_classes_are_mutually_exclusive_and_ordered_by_reachability():
    assert primary_failure(True, True, 1, True) == "success"
    assert primary_failure(False, True, 1, True) == "entered_20m_then_left"
    assert primary_failure(False, False, None, False) == "no_correct_candidate_in_spatial_nms20"
    assert primary_failure(False, False, 3, False) == "nms_top5_contains_target_but_policy_never_selected_it"
    assert primary_failure(False, False, 3, True) == "nms_top5_contains_target_but_trajectory_did_not_reach_it"
    assert primary_failure(False, False, 9, False) == "correct_candidate_only_at_nms_ranks6_20"


def test_relation_group_is_explicitly_only_a_rule_based_proxy():
    assert relation_proxy("The car north of The Store", ["The Store"]) == "direct_relation_word_proxy"
    assert relation_proxy("The car in the parking lot near The Store", ["The Store"]) == "indirect_or_multi-entity_proxy"
    assert relation_proxy("Find the car", []) == "unbound_or_unmentioned"
