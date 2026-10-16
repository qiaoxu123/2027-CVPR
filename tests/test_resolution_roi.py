import cv2
import numpy as np
from graphnav.observed_patch_scales import observed_region_masks


def test_cropped_support_mask_preserves_full_canvas_rasterization():
    rng=np.random.default_rng(0)
    support=rng.integers(0,2,(384,384)).astype(bool)
    for angle in np.linspace(0,6,20):
        c,s=np.cos(angle),np.sin(angle)
        polygon=np.rint(np.array([[-50,-35],[50,-35],[50,35],[-50,35]])@np.array([[c,s],[-s,c]])+192).astype(np.int32)
        full=np.zeros((384,384),np.uint8);cv2.fillConvexPoly(full,polygon,1)
        lo=polygon.min(0);hi=polygon.max(0)+1
        local=np.zeros((hi[1]-lo[1],hi[0]-lo[0]),np.uint8);cv2.fillConvexPoly(local,polygon-lo,1)
        assert np.array_equal(full[lo[1]:hi[1],lo[0]:hi[0]],local)
        assert support[full.astype(bool)].mean()==support[lo[1]:hi[1],lo[0]:hi[0]][local.astype(bool)].mean()
