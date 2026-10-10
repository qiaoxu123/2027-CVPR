import numpy as np
import pytest
import torch
from graphnav.appearance_matching import MultiScaleAppearanceMatcher, ObservedSigLIPMatchingEncoder


def test_initial_single_scale_is_native_cosine_and_unknown_is_not_negative():
    torch.manual_seed(2)
    model = MultiScaleAppearanceMatcher(dimension=8, hidden=4)
    text = torch.randn(2,8)
    images = torch.randn(2,3,1,8)
    valid = torch.ones(2,3,1,dtype=torch.bool)
    valid[:,2] = False
    scores,known,weights = model(text,images,valid,torch.zeros(2,3,1,3))
    expected = torch.nn.functional.cosine_similarity(text[:,None],images[:,:,0],dim=-1)
    assert torch.allclose(scores[:,:2],expected[:,:2]*model.log_scale.exp(),atol=1e-5)
    assert torch.equal(scores[:,2],torch.zeros(2))
    assert not known[:,2].any()
    assert not weights[:,2].any()


def test_invalid_scales_do_not_change_score_or_gradient():
    model = MultiScaleAppearanceMatcher(dimension=8,hidden=4)
    text = torch.randn(1,8)
    image = torch.randn(1,2,2,8,requires_grad=True)
    valid = torch.tensor([[[True,False],[True,True]]])
    score,_,_ = model(text,image,valid,torch.zeros(1,2,2,3))
    modified = image.detach().clone();modified[0,0,1] = 1000
    again,_,_ = model(text,modified,valid,torch.zeros(1,2,2,3))
    assert torch.allclose(score,again)
    score.sum().backward()
    assert image.grad[0,0,1].count_nonzero()==0
    assert model.image_residual[-1].weight.grad.abs().sum()>0


def test_cache_key_changes_with_actual_pixels_or_native_dimensions():
    x=np.full((384,384,3),120,np.uint8);y=x.copy();y[0,0]=121
    assert ObservedSigLIPMatchingEncoder.image_key(x)!=ObservedSigLIPMatchingEncoder.image_key(y)
    assert ObservedSigLIPMatchingEncoder.image_key(x)!=ObservedSigLIPMatchingEncoder.image_key(x[:32,:32])
    with pytest.raises(ValueError):ObservedSigLIPMatchingEncoder.image_key(x.astype(float))


def test_native_text_uses_model_position_limit_and_caches_actual_queries():
    from types import SimpleNamespace
    class Batch(dict):
        def to(self, device): return self
    calls=[]
    def processor(**kwargs):
        calls.append(kwargs)
        assert kwargs['max_length']==64
        return Batch(input_ids=torch.zeros(len(kwargs['text']),64,dtype=torch.long))
    encoder=ObservedSigLIPMatchingEncoder.__new__(ObservedSigLIPMatchingEncoder)
    encoder.device=torch.device('cpu');encoder.text_cache={};encoder.processor=processor
    encoder.model=SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=64)),
                                  get_text_features=lambda **kwargs: torch.ones(len(kwargs['input_ids']),8))
    first=encoder.texts(['brown '*100,'gray roof'])
    second=encoder.texts(['gray roof','brown '*100])
    assert len(calls)==1
    assert torch.equal(first.flip(0),second)
