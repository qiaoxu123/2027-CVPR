from types import SimpleNamespace
import numpy as np
import torch
from experiments.a2_waldo_landmark_belief.perception import FrozenBoxCropEncoder


def test_tiny_box_crops_have_unambiguous_rgb_channel_layout():
    class Batch(dict):
        def to(self,device):
            return self
    def processor(images,return_tensors,input_data_format):
        assert all(image.mode=='RGB' for image in images)
        assert input_data_format=='channels_last'
        return Batch(pixel_values=torch.zeros(len(images),3,8,8))
    native=SimpleNamespace(model=SimpleNamespace(get_image_features=lambda **x:torch.ones(len(x['pixel_values']),16)),
                           processor=processor,device=torch.device('cpu'),dimension=16)
    encoder=FrozenBoxCropEncoder(native)
    result=encoder.images([np.zeros((3,7,3),np.uint8),np.zeros((7,3,3),np.uint8)])
    assert result.shape==(2,16) and torch.isfinite(result).all()
