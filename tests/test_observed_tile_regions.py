import numpy as np
import pytest
from graphnav.observed_tile_regions import split_current_tiles,tile_region_inputs
from graphnav.local_chain_observation import local_patch_regions


def views():
    quad=np.array([[0,0],[96,0],[96,96],[0,96]],np.float64)
    low=dict(bgr=np.full((384,384,3),80,np.uint8),depth_valid=np.ones((384,384),bool),footprint_world=quad,pose=np.array([48,48,50,0]))
    high=dict(bgr=np.full((768,768,3),80,np.uint8),depth_valid=np.ones((768,768),bool),footprint_world=quad.copy(),pose=np.array([48,48,50,0]))
    return low,high


def test_actual_tiles_preserve_quadrant_order_and_input_shape():
    _,high=views()
    for i,(y,x) in enumerate([(0,0),(0,384),(384,0),(384,384)]):high['bgr'][y:y+384,x:x+384]=i
    tiles=split_current_tiles(high['bgr'])
    assert [int(t[0,0,0]) for t in tiles]==[0,1,2,3]
    assert all(t.shape==(384,384,3) for t in tiles)
    with pytest.raises(ValueError):split_current_tiles(high['bgr'][:384,:384])


def test_sparse_sites_equal_original_world_geometry_and_subset_of_dense():
    low,high=views();sparse=tile_region_inputs(low,high);dense=tile_region_inputs(low,high,dense=True)
    original=local_patch_regions(low)
    assert np.allclose(sparse['world'],original[0],atol=1e-5)
    assert np.allclose(sparse['quality'],original[3])
    ids=np.array([y*24+x for y in range(1,24,2) for x in range(1,24,2)])
    assert np.array_equal(sparse['masks'],dense['masks'][ids])
    assert sparse['masks'].shape==(144,3,2304)
    assert dense['masks'].shape==(576,3,2304)
    assert (sparse['masks'][:,0].sum(-1)==4).all()


def test_shared_validity_rejects_unsupported_pixels_and_pose_changes():
    low,high=views();high['depth_valid'][:]=False
    result=tile_region_inputs(low,high)
    assert not result['valid'].any() and not result['masks'].any()
    _,high=views();high['pose']=np.array([49,48,50,0])
    with pytest.raises(ValueError):tile_region_inputs(low,high)


def test_private_target_fields_do_not_change_tiles_masks_or_world_sites():
    low,high=views();first=tile_region_inputs(low,high)
    high.update(target_world=[0,0],target_id=1);low.update(target_world=[90,90],target_id=500)
    second=tile_region_inputs(low,high)
    for key in ('world','masks','valid','quality'):assert np.array_equal(first[key],second[key])
