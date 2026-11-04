import torch
from graphnav.geometry_guided_zero_runtime import zero_rgb_feature_hook


def test_only_rgb_features_are_destroyed_without_geometry_or_language_changes():
    rgb=torch.randn(1,144,3,8);name=torch.randn(1,8);valid=torch.ones(1,144,dtype=torch.bool)
    original=dict(image_features=rgb,name_text=name,type_text=name,valid=valid)
    args,changed=zero_rgb_feature_hook(None,(),original)
    assert args==() and not changed['image_features'].any()
    assert torch.equal(original['image_features'],rgb) and original['image_features'].any()
    assert changed['name_text'] is name and changed['valid'] is valid


def test_spatial_shuffle_preserves_features_invalid_sites_and_score_multiset():
    from graphnav.shuffled_landmark_regression import shuffle_rgb_feature_hook
    from graphnav.geometry_guided_visual import LandmarkVisualBelief
    model=LandmarkVisualBelief(dimension=8,hidden=4)
    with torch.no_grad():model.score[-1].weight.normal_()
    image=torch.randn(1,4,3,8);valid=torch.tensor([[True,False,True,True]]);text=torch.randn(1,8)
    original=dict(image_features=image,name_text=text,type_text=text,valid=valid)
    _,changed=shuffle_rgb_feature_hook(None,(),original)
    assert torch.equal(changed['image_features'][:,1],image[:,1])
    a=model(**original);b=model(**changed)
    assert torch.allclose(a[valid].sort().values,b[valid].sort().values,atol=1e-6)
