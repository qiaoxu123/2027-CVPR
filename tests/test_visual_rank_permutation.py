import pytest
import torch
from graphnav.visual_rank_permutation import native_rank_permutation,weak_box_oracle_permutation,replace_visual_world_field
from graphnav.visual_chain_position import WorldLikelihoodPositionHead


def test_magnitude_multiset_and_invalid_sites_preserved_without_gt():
    old=torch.tensor([3.,-1.,2.,9.]);native=torch.tensor([.1,.8,.3,.9]);valid=torch.tensor([True,True,True,False])
    new=native_rank_permutation(old,native,valid)
    assert torch.equal(new[valid].sort().values,old[valid].sort().values)
    assert new.tolist()==[-1.,3.,2.,9.]
    with pytest.raises(PermissionError):weak_box_oracle_permutation(old,valid,valid)


def test_linear_world_intervention_matches_actual_frozen_head_hook():
    torch.manual_seed(0);model=WorldLikelihoodPositionHead(dimension=8,hidden=4).eval();count=6
    x=dict(prior_logits=torch.randn(1,16),local_uv=torch.rand(1,count,2),image_features=torch.randn(1,count,3,8),target_text=torch.randn(1,8),context_text=torch.randn(1,8),public_geometry=torch.randn(1,count,31),quality=torch.ones(1,count,6),local_valid=torch.tensor([[True,True,True,True,True,False]]))
    with torch.no_grad():old,visual=model(**x,return_visual=True)
    new=native_rank_permutation(visual[0],torch.arange(count).float(),x['local_valid'][0])
    changed=replace_visual_world_field(old[0,:16].log_softmax(-1),x['local_uv'][0],visual[0],new,x['local_valid'][0])
    handle=model.visual_score.register_forward_hook(lambda module,args,out:new[None,:,None])
    with torch.no_grad():direct=model(**x)
    handle.remove()
    assert torch.allclose(changed,direct[0,:16].log_softmax(-1),atol=1e-6)
    unchanged=replace_visual_world_field(old[0,:16].log_softmax(-1),x['local_uv'][0],visual[0],visual[0],x['local_valid'][0])
    assert torch.allclose(unchanged,old[0,:16].log_softmax(-1),atol=1e-7)
