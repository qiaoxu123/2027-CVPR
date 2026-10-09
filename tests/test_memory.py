import numpy as np

from graphnav.memory import ObservedRGBMemory
from graphnav.foundations import CandidateVisionCache


def test_causal_mosaic_is_empty_until_observed():
    m = ObservedRGBMemory((0, 0, 400, 400), size=256)
    unseen = m.crop((320, 320), extent_m=60, size=32)
    assert unseen.coverage == 0
    assert np.all(unseen.image == 0)
    assert m.explored_fraction == 0
    view = np.full((48, 48, 3), (255, 80, 10), dtype=np.uint8)
    corner = m.view_corners((80, 80), 0, 30)
    m.observe(view, corner)
    inside = m.crop((80, 80), extent_m=40, size=32)
    assert inside.coverage > 0.9
    assert inside.center_coverage > 0.9
    assert inside.image[16, 16, 0] > 200
    still_unseen = m.crop((320, 320), extent_m=60, size=32)
    assert still_unseen.coverage == 0
    assert not still_unseen.image.any()
    masks = m.masks(28)
    assert masks.shape == (2, 28, 28)


def test_vis_cache_does_not_confuse_seen_and_unseen():
    cache = CandidateVisionCache(limit=2)
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    mask0 = np.zeros((16, 16), dtype=bool)
    mask1 = mask0.copy()
    mask1[0, 0] = True
    cache.put(img, mask0, __import__("torch").ones(4))
    assert cache.get(img, mask1) is None
