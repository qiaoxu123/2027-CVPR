import torch
from experiments.a2_waldo_landmark_belief.revision_model import RevisionNetwork
from experiments.a2_waldo_landmark_belief.revision_data import revision_labels
from experiments.a2_waldo_landmark_belief.collected_snapshot import snapshot_partition
from experiments.a2_waldo_landmark_belief.data_protocol import split_target_groups


def test_verified_cross_category_negatives_allowed_unknown_and_other_targets_masked():
    target=next(i for i in range(1000) if split_target_groups([dict(map_name='map',object_id=i)])[0]=='train')
    part=snapshot_partition('map',target)
    same=next(i for i in range(1000) if i!=target and snapshot_partition('map',i)==part and split_target_groups([dict(map_name='map',object_id=i)])[0]=='train')
    other=next(i for i in range(1000) if snapshot_partition('map',i)!=part)
    row=dict(target_instance=['map',target],private=dict(target_visible=True,target_kind=0,association=[
        dict(object_id=target,trusted=True),dict(object_id=same,trusted=True),dict(object_id=other,trusted=True),dict(object_id=None,trusted=False)]),
        public=dict(spatial=dict(grid_world=torch.tensor([[0.,0.],[50.,0.]]),candidate_xy=torch.tensor([[0.,0.],[50.,0.],[50.,0.],[0.,0.]]),candidate_support=torch.tensor([[1,0],[0,1],[0,1],[0,0]],dtype=torch.bool))),
        labels=dict(target_xy=torch.zeros(2),ranking=torch.tensor([1.,0,0,0]),rank_label_mask=torch.tensor([True,False,False,False]),offset_valid=torch.zeros(4,dtype=torch.bool)))
    labels=revision_labels(row,part,reserved=set())
    assert labels['rank_label_mask'].tolist()==[True,True,False,False]
    assert labels['offset_valid'].tolist()==[True,False,False,False]
    labels=revision_labels(row,part,reserved={('map',same)})
    assert labels['rank_label_mask'].tolist()==[True,False,False,False]


def inputs():
    return dict(instruction_tokens=torch.randn(2,4,12),token_valid=torch.ones(2,4,dtype=torch.bool),
      mention_mask=torch.ones(2,2,4,dtype=torch.bool),landmark_name_tokens=torch.randn(2,2,3,12),
      name_valid=torch.ones(2,2,3,dtype=torch.bool),link_valid=torch.ones(2,2,dtype=torch.bool),
      box_rgb=torch.randn(2,3,16),box_valid=torch.ones(2,3,dtype=torch.bool),
      pair_geometry=torch.randn(2,3,2,14),pair_valid=torch.ones(2,3,2,dtype=torch.bool),offset_limit=torch.ones(2,3,2)*4,
      a0_context=torch.rand(2,3,6),box_kind=torch.tensor([[0,1,0],[1,0,1]]))


def test_new_network_conditions_on_a0_and_has_zero_offsets_and_finite_gradients():
    torch.manual_seed(0);model=RevisionNetwork(text_dim=12,vision_dim=16,hidden=16,heads=2)
    x=inputs();out=model(**x)
    assert out['scores'].shape==(2,3) and out['category_logits'].shape==(2,2)
    assert (out['gate']>=0).all() and (out['gate']<=.25).all()
    assert out['offset'].count_nonzero()==0
    loss=out['scores'].sum()+out['candidate_match_logits'].sum()+out['benefit_logit'].sum()
    loss.backward()
    for prefix in ('base.language.','rank_context.','trust_context.','benefit.','category.'):
      assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for n,p in model.named_parameters() if n.startswith(prefix))
    old=out['benefit_logit'].detach();x['a0_context']=torch.zeros_like(x['a0_context']);new=model(**x)
    assert not torch.equal(old,new['benefit_logit'])


def test_invalid_landmarks_disable_gate():
    model=RevisionNetwork(text_dim=12,vision_dim=16,hidden=16,heads=2);x=inputs();x['link_valid'].zero_()
    out=model(**x);assert out['gate'].count_nonzero()==0 and not out['reliable'].any()


def test_zero_offset_trust_uses_kernel_peak_not_any_target_cell_in_large_box():
    target=next(i for i in range(1000) if split_target_groups([dict(map_name='map',object_id=i)])[0]=='train')
    part=snapshot_partition('map',target)
    row=dict(target_instance=['map',target],private=dict(target_visible=True,target_kind=1,association=[dict(object_id=target,trusted=True)]),
      public=dict(spatial=dict(grid_world=torch.tensor([[0.,0.],[50.,0.]]),candidate_xy=torch.tensor([[50.,0.]]),candidate_support=torch.ones(1,2,dtype=torch.bool))),
      labels=dict(target_xy=torch.zeros(2),ranking=torch.ones(1),rank_label_mask=torch.ones(1,dtype=torch.bool),offset_valid=torch.ones(1,dtype=torch.bool)))
    labels=revision_labels(row,part,reserved=set())
    assert not labels['candidate_trust'].any()


def test_private_target_fields_cannot_be_forward_inputs():
    import pytest
    model=RevisionNetwork(text_dim=12,vision_dim=16,hidden=16,heads=2)
    with pytest.raises(TypeError):model(**inputs(),target_xy=torch.zeros(2,2))
