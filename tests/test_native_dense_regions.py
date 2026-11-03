import numpy as np
from graphnav.local_chain_observation import local_patch_regions
from graphnav.native_dense_regions import native_dense_regions,sparse_view,SPARSE_IDS


def test_native144_is_exact_dense_subset_with_partial_support():
    rgb=np.ones((384,384,3),np.uint8)*100;depth=np.ones((384,384),bool)
    depth[32:64,48:85]=False;rgb[:16,:32]=0
    obs=dict(bgr=rgb,depth_valid=depth,footprint_world=[[100,200],[100,120],[20,120],[20,200]])
    dense=native_dense_regions(obs);sparse=sparse_view(dense);world,masks,valid,quality=local_patch_regions(obs)
    assert dense['world'].shape==(576,2) and len(SPARSE_IDS)==144
    assert np.array_equal(sparse['world'],world)
    assert np.array_equal(sparse['masks'],masks)
    assert np.array_equal(sparse['valid'],valid)
    assert np.allclose(sparse['quality'],quality,atol=1e-7)
    assert not dense['masks'][~dense['valid']].any()
