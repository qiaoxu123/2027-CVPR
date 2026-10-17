import torch
from graphnav.geombelief_visual_inputs import residual_input_mask

def test_geometry_pose_has_no_rgb_descriptors_or_scores():
    x=torch.randn(2,784,106);y=residual_input_mask(x,'geometry_pose')
    assert not y[...,16:80].any() and not y[...,100:102].any()
    assert torch.equal(x[...,80:98],y[...,80:98])
    assert torch.equal(x[...,:16],y[...,:16]) and torch.equal(x[...,104],y[...,104])

def test_rgb_text_has_no_geometry_logits_or_pose():
    x=torch.randn(2,784,106);y=residual_input_mask(x,'rgb_text')
    assert not y[...,:16].any() and not y[...,98:100].any() and not y[...,104].any()
    assert torch.equal(x[...,16:98],y[...,16:98])
