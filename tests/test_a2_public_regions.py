"""Exact public landmark surfaces, independent of private training targets."""
import json

import numpy as np
import pytest

from experiments.a2_waldo_landmark_belief.public_regions import (
    region_from_contours, region_shape, region_statistics,
)


def square(x, y, size=1):
    return [[x, y], [x+size, y], [x+size, y+size], [x, y+size]]


def test_overlap_union_does_not_double_count_area_and_is_order_independent():
    contours = [square(0, 0, 2), square(1, 0, 2)]
    region = region_from_contours(contours)
    assert region_shape(region).area == pytest.approx(6)
    assert region_statistics(region, [1.5, 1])['center'] == pytest.approx([1.5, 1])
    assert region_shape(region).equals(region_shape(region_from_contours(contours[::-1])))
    assert region_shape(json.loads(json.dumps(region))).equals(region_shape(region))


def test_concave_gap_is_outside_and_distance_uses_actual_boundary():
    region = region_from_contours([square(0, 0), square(1, 0), square(0, 1)])
    info = region_statistics(region, [1.5, 1.5])
    assert not info['inside']
    assert info['signed_distance'] == pytest.approx(-.5)
    boundary = region_statistics(region, [1, 1.5])
    assert boundary['inside'] and boundary['signed_distance'] == 0


def test_hole_remains_outside_after_union_and_serialization():
    # Four rectangular surfaces surround a genuine 1x1 empty square.
    strips = [[[0,0],[3,0],[3,1],[0,1]], [[0,2],[3,2],[3,3],[0,3]],
              [[0,1],[1,1],[1,2],[0,2]], [[2,1],[3,1],[3,2],[2,2]]]
    region = json.loads(json.dumps(region_from_contours(strips)))
    info = region_statistics(region, [1.5, 1.5])
    assert info['area'] == pytest.approx(8)
    assert not info['inside'] and info['signed_distance'] == pytest.approx(-.5)
    assert len(region_shape(region).interiors) == 1


def test_point_touch_is_valid_multipolygon_without_filling_empty_space():
    region = region_from_contours([square(0, 0), square(1, 1)])
    assert region_shape(region).geom_type == 'MultiPolygon'
    assert region_statistics(region, [.5, 1.5])['signed_distance'] == pytest.approx(-.5)
    assert region_shape(region).area == pytest.approx(2)


@pytest.mark.parametrize('contours', [[], [[]], [[[0,0],[1,1]]],
    [[[0,0],[1,1],[2,2]]], [[[0,0],[1,1],[0,1],[1,0]]],
    [[[0,0],[float('nan'),0],[1,1]]], [[[0,0],[float('inf'),0],[1,1]]],
    [[[0,0,1],[1,0,1],[1,1,1]]]])
def test_invalid_member_is_rejected_instead_of_repaired(contours):
    with pytest.raises(ValueError):
        region_from_contours(contours)


def test_invalid_serialized_region_and_nonfinite_points_are_rejected():
    region = region_from_contours([square(0, 0)])
    with pytest.raises(ValueError):
        region_shape(dict(region, version='unknown'))
    with pytest.raises(ValueError):
        region_statistics(region, [np.nan, 0])
    with pytest.raises(ValueError):
        region_statistics(region, [0, 1, 2])
    with pytest.raises(ValueError):
        region_shape(dict(region, geometry={'type':'Point','coordinates':[0,0]}))


def test_input_arrays_are_unchanged():
    contour = np.asarray(square(0, 0), dtype=np.float32)
    before = contour.copy()
    region_from_contours([contour])
    np.testing.assert_array_equal(contour, before)


@pytest.mark.parametrize('geometry', [None, [], {}, {'type': 'Unsupported'},
                                      {'type':'Polygon','coordinates':[[[0,0],[1,0]]]}])
def test_malformed_serialized_shapes_fail_with_a_clear_validation_error(geometry):
    with pytest.raises(ValueError):
        region_shape(dict(version='public-region-v1', geometry=geometry))
