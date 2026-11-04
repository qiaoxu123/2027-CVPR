"""Frozen diagnostic control, not a normal inference option."""
import torch
from .geometry_guided_runtime import GeometryGuidedLandmarkPolicy


def zero_rgb_feature_hook(module,args,kwargs):
    changed=dict(kwargs)
    changed['image_features']=torch.zeros_like(kwargs['image_features'])
    return args,changed


class ZeroFeatureLandmarkPolicy(GeometryGuidedLandmarkPolicy):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.zero_hook=self.learned_visual.register_forward_pre_hook(zero_rgb_feature_hook,with_kwargs=True)
