import numpy as np
from graphnav.proposal_grid_audit import observed_grid_sites
from graphnav.local_chain_observation import local_patch_regions


def observation():
    return dict(bgr=np.ones((384,384,3),np.uint8)*100, depth_valid=np.ones((384,384),bool),
                footprint_world=np.array([[0,0],[383,0],[383,383],[0,383]],np.float32))


def test_existing_grid_is_reproduced_and_dense_grid_contains_it():
    obs=observation(); old,_,valid,_=local_patch_regions(obs)
    sites, mask=observed_grid_sites(obs,2)
    np.testing.assert_allclose(sites,old);assert np.array_equal(mask,valid)
    dense,_=observed_grid_sites(obs,1)
    assert dense.shape==(576,2)
    assert all(np.linalg.norm(dense-point,axis=1).min()<1e-5 for point in sites)


def test_invalid_rgb_or_depth_is_not_a_supported_site():
    obs=observation();obs['bgr'][16:32,16:32]=0
    _,valid=observed_grid_sites(obs,2)
    assert not valid[0]
