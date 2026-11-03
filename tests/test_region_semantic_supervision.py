import numpy as np
from graphnav.region_semantic_supervision import semantic_region_labels


def square(x, y, size=10):
    return [[x, y], [x+size, y], [x+size, y+size], [x, y+size]]


def test_separate_regions_are_labelled_but_unknown_is_not_background():
    patches = np.array([square(3, 3, 2), square(23, 3, 2), square(43, 3, 2), square(70, 70, 2)])
    contours = {'Building': [square(0, 0)], 'Parking': [square(20, 0)], 'Ground': [square(40, 0)]}
    labels, _ = semantic_region_labels(patches, contours, np.ones(4, bool), margin_m=1)
    assert labels.tolist() == [0, 1, 2, -1]


def test_cross_class_overlap_and_boundary_abstain():
    contours = {'Building': [square(0, 0)], 'Parking': [square(0, 0)], 'Ground': []}
    patches = np.array([square(3, 3, 2), square(0, 0, 2)])
    assert (semantic_region_labels(patches, contours, np.ones(2, bool), margin_m=1)[0] == -1).all()


def test_missing_and_invalid_regions_do_not_create_false_labels():
    patches = np.array([square(3, 3, 2)])
    contours = {'Building': [[[0, 0], [1, 1]]], 'Parking': [], 'Ground': []}
    assert semantic_region_labels(patches, contours, np.ones(1, bool))[0].tolist() == [-1]
