import numpy as np

from graphnav.public_reference_supervision import reference_region_labels, patch_world_windows


def square(x, y, side=2):
    return [[x, y], [x+side, y], [x+side, y+side], [x, y+side]]


def test_core_boundary_and_distant_regions_are_distinct():
    windows = np.array([square(3, 3), square(0, 0), square(20, 20)], float)
    labels, coverage = reference_region_labels(windows, square(0, 0, 10),
        np.ones(3, bool), margin_m=1)
    assert labels.tolist() == [1, -1, 0]
    assert coverage[0, 0] == 1 and coverage[2, 1] == 0


def test_invalid_polygon_and_invalid_depth_abstain():
    windows = np.array([square(3, 3), square(20, 20)], float)
    assert (reference_region_labels(windows, [[0, 0], [1, 1]], np.ones(2, bool))[0] == -1).all()
    assert (reference_region_labels(windows, square(0, 0, 10), np.zeros(2, bool))[0] == -1).all()


def test_small_region_eroded_away_has_no_positive():
    labels, _ = reference_region_labels(np.array([square(0, 0, 1)]),
        square(0, 0, 1), np.ones(1, bool), margin_m=2)
    assert labels.tolist() == [-1]


def test_patch_support_reproduces_border_clipping_and_yaw():
    footprint = np.array([[0, 0], [383, 0], [383, 383], [0, 383]], float)
    windows = patch_world_windows(footprint)
    assert windows.shape == (144, 3, 4, 2)
    np.testing.assert_allclose(windows[-1, 2].max(0), [383, 383])
    assert (windows[0, 2].min(0) == 0).all()
    turned = patch_world_windows(footprint@np.array([[0, 1], [-1, 0]]))
    np.testing.assert_allclose(turned, windows@np.array([[0, 1], [-1, 0]]), atol=1e-4)


def test_no_positive_does_not_imply_reference_absence():
    labels, _ = reference_region_labels(np.array([square(30, 30)]),
        square(0, 0, 10), np.ones(1, bool))
    assert labels.tolist() == [0]
    # Caller must require positive AND negative in-view support for ranking;
    # these zeros are geometrical negatives, not a Null/absence target.
    assert not ((labels == 1).any() and (labels == 0).any())
