import numpy as np
import pytest
from graphnav.observed_candidate_crops import observed_candidate_crops


def test_crops_are_exact_slices_of_current_rgb_and_stay_in_view():
    rgb=np.arange(384*384*3,dtype=np.int64).reshape(384,384,3).astype(np.uint8)
    crops,meta=observed_candidate_crops(rgb,np.array([[23.5,23.5],[375.5,375.5]]),np.array([True,True]),(48,112))
    assert len(crops)==4
    for crop,record in zip(crops,meta):
        x0,y0,x1,y1=record['box_xyxy'];np.testing.assert_array_equal(crop,rgb[y0:y1,x0:x1]);assert 0<=x0<x1<=384 and 0<=y0<y1<=384
    assert meta[-1]['clipped']


def test_invalid_candidates_do_not_produce_fake_crops():
    crops,meta=observed_candidate_crops(np.zeros((384,384,3),np.uint8),np.array([[20.,20.]]),np.array([False]),(48,))
    assert crops==[] and meta==[]
    with pytest.raises(ValueError):
        observed_candidate_crops(np.zeros((800,800,3),np.uint8),np.array([[20.,20.]]),np.array([True]),(48,))
