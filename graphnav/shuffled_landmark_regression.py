"""Destroy image/world correspondence, preserve valid feature multiset."""
from .geometry_guided_runtime import GeometryGuidedLandmarkPolicy


def shuffle_rgb_feature_hook(module,args,kwargs):
    changed=dict(kwargs);image=kwargs['image_features'].clone();valid=kwargs['valid']
    for i in range(len(image)):
        ids=valid[i].nonzero().flatten();image[i,ids]=image[i,ids.flip(0)].clone()
    changed['image_features']=image
    return args,changed


class ShuffledLandmarkRegressionPolicy(GeometryGuidedLandmarkPolicy):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.control_hook=self.learned_visual.register_forward_pre_hook(shuffle_rgb_feature_hook,with_kwargs=True)
