import numpy as np
from graphnav.local_chain_observation import local_patch_regions
from graphnav.memory import ObservedRGBMemory


def observation(valid=True):
    return dict(bgr=np.full((384,384,3),128,np.uint8),
                depth_valid=np.full((384,384),valid,bool),
                footprint_world=ObservedRGBMemory.view_corners([30.,40.],0.,10.))


def test_local_hypotheses_exist_in_small_low_altitude_view():
    xy,masks,valid,quality=local_patch_regions(observation())
    assert xy.shape==(144,2) and masks.shape==(144,3,576)
    assert valid.all() and quality.shape==(144,6)
    assert np.max(np.abs(xy-[30,40]))<10.
    assert masks[:,0].sum(-1).min()==1


def test_unknown_depth_does_not_make_visual_hypotheses():
    xy,masks,valid,quality=local_patch_regions(observation(False))
    assert not valid.any() and not masks.any()
    assert np.isfinite(quality).all()
