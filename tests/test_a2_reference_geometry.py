import numpy as np
import pytest

from experiments.a2_waldo_landmark_belief.core import box_landmark_geometry, _center
from experiments.a2_waldo_landmark_belief.public_regions import region_from_contours
from experiments.a2_waldo_landmark_belief.reference_geometry import (
    public_reference_center, box_reference_geometry,
)


def square(x=0, y=0, side=1):
    return [[x,y],[x+side,y],[x+side,y+side],[x,y+side]]


def group(contours):
    return dict(region=region_from_contours(contours), height_policy='missing_for_public_road_group')


def test_singleton_features_are_exact_legacy_outputs_including_height():
    boxes, refs = [square(2,2),square(5,1)], [dict(contour=square()),dict(contour=square(4,3))]
    old = box_landmark_geometry(boxes,[r['contour'] for r in refs],[.7,.8],
                               box_height_m=[10,None],landmark_height_m=[3,4])
    new = box_reference_geometry(boxes,refs,[.7,.8],box_height_m=[10,None],landmark_height_m=[3,4])
    for a,b in zip(old,new): np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(public_reference_center(refs[0]),_center(np.array(square(),np.float32)))


def test_group_features_use_union_centroid_area_boundary_and_missing_height():
    ref = group([square(),square(1,0)])
    feature, valid = box_reference_geometry([square(2,0)],[ref],[.9],box_height_m=[100],landmark_height_m=[20])
    assert valid.tolist() == [[True]]
    assert public_reference_center(ref) == pytest.approx([1,.5])
    assert feature[0,0,0] == pytest.approx(1.5/40)
    assert feature[0,0,1] == 0
    assert feature[0,0,2] == pytest.approx(1.5/40)
    assert feature[0,0,6] == pytest.approx(-.5/40)
    assert feature[0,0,8] == pytest.approx(np.log(.5)/5)
    assert feature[0,0,10:].tolist() == [0,0]
    missing, _ = box_reference_geometry([square(2,0)],[ref],[.9],box_height_m=[None],landmark_height_m=[None])
    np.testing.assert_array_equal(feature,missing)


def test_union_hole_is_outside_even_if_it_contains_union_centroid():
    strips = [[[0,0],[3,0],[3,1],[0,1]],[[0,2],[3,2],[3,3],[0,3]],
              [[0,1],[1,1],[1,2],[0,2]],[[2,1],[3,1],[3,2],[2,2]]]
    feature,valid = box_reference_geometry([square(1.25,1.25,.5)],[group(strips)],[1])
    assert valid.all() and feature[0,0,7] == 0
    assert feature[0,0,6] == pytest.approx(-.5/40)


def test_invalid_boxes_or_confidence_cannot_create_valid_relationships():
    feature,valid = box_reference_geometry([[[0,0],[1,1]],square()],[group([square()])],[.9,float('nan')])
    assert not valid.any() and not feature.any()


def test_empty_axes_and_alignment_validation_match_legacy():
    feature,valid = box_reference_geometry([],[],[])
    assert feature.shape == (0,0,12) and valid.shape == (0,0)
    with pytest.raises(ValueError,match='confidence count'):
        box_reference_geometry([square()],[group([square()])],[])
    with pytest.raises(ValueError,match='height count'):
        box_reference_geometry([square()],[group([square()])],[1],landmark_height_m=[])


def test_group_cannot_claim_observed_height_and_bad_serialization_fails():
    ref = group([square()])
    ref['height_policy'] = 'invented'
    with pytest.raises(ValueError,match='height policy'):
        box_reference_geometry([square()],[ref],[1])
    with pytest.raises(ValueError): public_reference_center(dict(region={'version':'unknown'}))
