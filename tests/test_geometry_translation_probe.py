import torch
from graphnav.geometric_translation_probe import translate_observable, average_aligned_logits
from graphnav.full_localization import FullTokenGeometryLocalizer


def test_translation_preserves_relative_fields_and_padding():
    batch=dict(grid_xy=torch.tensor([[[.1,.2],[.9,.8]]]),
               nodes=torch.zeros(1,2,10),node_valid=torch.tensor([[True,False]]),
               pair_geometry=torch.randn(1,2,2,10))
    moved,valid=translate_observable(batch,(.2,-.1))
    torch.testing.assert_close(moved['grid_xy'],torch.tensor([[[.3,.1],[1.1,.7]]]))
    assert valid.tolist()==[[True,False]]
    torch.testing.assert_close(moved['nodes'][0,0,:2],torch.tensor([.2,-.1]))
    assert moved['nodes'][0,1].count_nonzero()==0
    torch.testing.assert_close(moved['pair_geometry'],batch['pair_geometry'])
    assert batch['nodes'].count_nonzero()==0


def test_center_relative_model_is_equivariant_for_paired_public_translation():
    torch.manual_seed(42);torch.set_num_threads(2)
    model=FullTokenGeometryLocalizer(hidden=16,text_dim=8,heads=2,
                                    max_relations=2,top_k=2,dropout=0,
                                    relative_position=True).eval()
    batch=dict(prior=torch.randn(1,9),rules=torch.randn(1,9),
               relation_fields=torch.randn(1,9,2),grid_xy=torch.rand(1,9,2),
               nodes=torch.rand(1,3,10),node_types=torch.zeros(1,3,dtype=torch.long),
               node_valid=torch.ones(1,3,dtype=torch.bool),
               entity_valid=torch.ones(1,3,dtype=torch.bool),
               pair_geometry=torch.randn(1,9,3,10))
    moved,_=translate_observable(batch,(.1,-.1))
    tokens=torch.randn(1,5,8);mask=torch.ones(1,5,dtype=torch.bool);names=torch.randn(1,3,8)
    with torch.no_grad():
        a=model(batch,variant='geometry',tokens=tokens,token_mask=mask,node_emb=names)
        b=model(moved,variant='geometry',tokens=tokens,token_mask=mask,node_emb=names)
    torch.testing.assert_close(a.belief_logits,b.belief_logits,atol=1e-6,rtol=1e-5)


def test_aligned_average_preserves_constant_gauge_and_unknown_queries():
    base=torch.tensor([[1.,2.,3.,4.]])
    branches=[(base+10,torch.tensor([[True,True,False,False]])),
              (base-7,torch.tensor([[False,True,True,False]]))]
    result=average_aligned_logits(base,branches)
    torch.testing.assert_close(result,base)
    assert result[0,3]==base[0,3]
