import numpy as np
from graphnav.rgb_content_controls import controlled_rgb


def test_gray_removes_content_without_mutating_observation():
    image=np.arange(384*384*3,dtype=np.uint32).reshape(384,384,3).astype(np.uint8)
    original=image.copy();gray=controlled_rgb(image,'gray-rgb')
    np.testing.assert_array_equal(image,original)
    assert gray.shape==image.shape and gray.dtype==image.dtype
    assert (gray==128).all()


def test_patch_shuffle_preserves_all_pixel_content_and_is_deterministic():
    image=np.random.default_rng(7).integers(0,256,(384,384,3),dtype=np.uint8)
    a=controlled_rgb(image,'rgb-shuffle');b=controlled_rgb(image,'rgb-shuffle')
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(np.sort(a.reshape(-1)),np.sort(image.reshape(-1)))
    assert not np.array_equal(a,image)
    np.testing.assert_array_equal(controlled_rgb(image,'real'),image)
