import numpy as np
from graphnav.partial_observed_roi import observed_intersection_crop


def test_full_coverage_keeps_pixels_and_coordinates():
    image = np.arange(8*8*3,dtype=np.uint8).reshape(8,8,3)
    quad = np.float32([[0,0],[7,0],[7,7],[0,7]])
    crop, bounds, fraction = observed_intersection_crop(image, quad, quad)
    np.testing.assert_array_equal(image, crop)
    assert bounds == (0,0,8,8)
    assert fraction == 1.


def test_half_footprint_excludes_unobserved_pixels():
    image = np.full((8,8,3), 255, dtype=np.uint8)
    quad = np.float32([[0,0],[7,0],[7,7],[0,7]])
    camera = np.float32([[0,0],[3,0],[3,7],[0,7]])
    crop, bounds, fraction = observed_intersection_crop(image, quad, camera)
    assert bounds == (0,0,4,8)
    assert crop.shape == (8,4,3)
    assert fraction == .5
    # Input storage is never altered by an experimental crop.
    assert image[:,4:].min() == 255


def test_disjoint_observation_has_no_visual_crop():
    quad = np.float32([[0,0],[7,0],[7,7],[0,7]])
    crop, bounds, fraction = observed_intersection_crop(np.ones((8,8,3),np.uint8),quad,quad+20)
    assert crop is None and bounds is None and fraction == 0.


def test_rotated_footprint_masks_unknown_pixels_inside_crop_rectangle():
    quad = np.float32([[0,0],[7,0],[7,7],[0,7]])
    camera = np.float32([[3,0],[6,3],[3,6],[0,3]])
    crop, bounds, _ = observed_intersection_crop(np.full((8,8,3),255,np.uint8),quad,camera)
    assert bounds == (0,0,7,7)
    assert crop[0,0].max() == 0
    assert crop[3,3].min() == 255
