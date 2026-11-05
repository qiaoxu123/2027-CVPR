"""Pure CPU tests: public-Polygon candidate attention; no GT input."""
import torch
import pytest

from graphnav.polygon_language_attention import PolygonGuidedLanguageAttention


def fixture(batch=2,queries=12,names=3,length=7):
    torch.manual_seed(42)
    x=dict(
        base_logits=torch.randn(batch,queries),
        dense_context=torch.randn(batch,queries,128),
        pair_geometry=torch.randn(batch,queries,names,10),
        entity_valid=torch.tensor([[True,True,False],[False,False,False]]),
        node_emb=torch.randn(batch,names,768),
        tokens=torch.randn(batch,length,768),
        token_mask=torch.tensor([[True]*7,[True]*5+[False]*2]),
    )
    return x


def test_zero_init_exactly_preserves_base_and_no_anchor_fallback():
    model=PolygonGuidedLanguageAttention()
    x=fixture()
    out=model(**x,return_attention=True)
    assert torch.equal(out["logits"],x["base_logits"])
    assert torch.equal(out["delta"],torch.zeros_like(x["base_logits"]))
    assert out["supported"].tolist()==[True,False]
    assert out["token_weights"].shape==(2,12,7)
    assert out["entity_weights"].shape==(2,12,3)
    assert torch.equal(out["token_weights"][0,:,0],torch.zeros(12))
    assert torch.equal(out["token_weights"][0,:,6],torch.zeros(12))
    assert torch.equal(out["token_weights"][1,:,4],torch.zeros(12))
    assert out["token_weights"][0,:,1:6].sum(-1).allclose(torch.ones(12))
    assert out["entity_weights"][0,:,2].abs().max()==0
    assert out["entity_weights"][1].abs().max()==0


def test_learned_residual_responds_to_language_and_polygon_and_is_bounded():
    model=PolygonGuidedLanguageAttention()
    with torch.no_grad():
        # Enable nonzero output so sensitivity is exercised without GPU training.
        model.score[-1].weight.normal_(0,.05)
    x=fixture()
    out=model(**x,return_attention=True)
    assert (out["delta"].abs()<=.500001).all()
    assert torch.equal(out["logits"][1],x["base_logits"][1])
    assert (out["delta"][0].abs()>1e-8).any()
    changed=dict(x)
    changed["tokens"]=x["tokens"].clone()
    changed["tokens"][0,1]=changed["tokens"][0,1]*(-4)
    y=model(**changed,return_attention=True)
    assert not torch.allclose(out["logits"][0],y["logits"][0])
    altered=dict(x)
    altered["pair_geometry"]=x["pair_geometry"].clone()
    altered["pair_geometry"][0,:,:2,:]*=-1
    z=model(**altered,return_attention=True)
    assert not torch.allclose(out["logits"][0],z["logits"][0])


def test_only_adapter_gradients_are_needed_and_gt_cannot_be_passed():
    model=PolygonGuidedLanguageAttention()
    x=fixture()
    out=model(**x)
    label=torch.tensor([3,6])
    loss=torch.nn.functional.cross_entropy(out["logits"],label)
    loss.backward()
    assert model.score[-1].weight.grad is not None
    assert model.score[-1].weight.grad.abs().sum()>0
    with pytest.raises(TypeError):
        model(**x,target_xy=torch.zeros(2,2))


def test_all_text_special_only_abstains():
    model=PolygonGuidedLanguageAttention()
    x=fixture()
    x["token_mask"][:]=False
    x["token_mask"][:,0:2]=True
    out=model(**x,return_attention=True)
    assert not out["supported"].any()
    assert torch.equal(out["logits"],x["base_logits"])
    assert torch.equal(out["token_weights"],torch.zeros_like(out["token_weights"]))


def test_dimension_errors_are_not_silent():
    model=PolygonGuidedLanguageAttention()
    x=fixture()
    x["pair_geometry"]=x["pair_geometry"][:,:,:,:9]
    with pytest.raises(ValueError,match="Mismatched"):
        model(**x)
