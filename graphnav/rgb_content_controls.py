"""Inference-only RGB interventions. Geometry/masks must come from real RGB first."""
import numpy as np


def controlled_rgb(bgr,control):
    image=np.asarray(bgr)
    if image.shape!=(384,384,3) or image.dtype!=np.uint8:
        raise ValueError('actual 384px uint8 BGR required')
    if control=='gray-rgb':return np.full_like(image,128)
    if control=='rgb-shuffle':
        patches=image.reshape(24,16,24,16,3).transpose(0,2,1,3,4).reshape(576,16,16,3)
        permutation=np.random.default_rng(0).permutation(576)
        return np.ascontiguousarray(patches[permutation].reshape(24,24,16,16,3).transpose(0,2,1,3,4).reshape(384,384,3))
    return image
