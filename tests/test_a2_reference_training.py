import copy
import torch
import pytest

from experiments.a2_waldo_landmark_belief.relation_v3_data import build_training_row
from experiments.a2_waldo_landmark_belief.public_regions import region_from_contours
from experiments.a2_waldo_landmark_belief.reference_training import build_reference_training_row


def raw():
    return dict(labels=dict(target_xy=torch.tensor([10.,0.])),private=dict(association=[dict(trusted=True,object_id=999)]),
        meta=dict(candidates=[dict(kind=0)],references=[dict(contour=[[0,0],[2,0],[2,2],[0,2]])]),
        public=dict(network=dict(box_rgb=torch.ones(1,6),box_valid=torch.ones(1,dtype=torch.bool),
            instruction_tokens=torch.zeros(2,8),token_valid=torch.ones(2,dtype=torch.bool),
            landmark_name_tokens=torch.zeros(1,2,8),name_valid=torch.ones(1,2,dtype=torch.bool),
            link_valid=torch.ones(1,dtype=torch.bool),mention_mask=torch.ones(1,2,dtype=torch.bool),
            pair_geometry=torch.zeros(1,1,14),pair_valid=torch.ones(1,1,dtype=torch.bool),offset_limit=torch.zeros(1,2)),
            spatial=dict(base_logits=torch.zeros(2),candidate_xy=torch.zeros(1,2),
                         grid_world=torch.tensor([[0.,0.],[100.,0.]]),candidate_support=torch.ones(1,2,dtype=torch.bool))))


def test_singleton_supervision_matches_original_and_private_associations_stay_unused():
    source = raw()
    old = build_training_row(source,target_kind=0,real=True)
    new = build_reference_training_row(source,target_kind=0,real=True)
    for key in old['labels']: assert torch.equal(old['labels'][key],new['labels'][key])
    assert not new['labels']['identity_label_mask'].any()
    assert 'private' not in new and 'target_xy' not in new['public']['network']
    source['private']['association'][0]['object_id'] = 123
    again = build_reference_training_row(source,target_kind=0,real=True)
    for key in new['labels']: assert torch.equal(new['labels'][key],again['labels'][key])


def test_group_supervision_uses_exact_public_union_centroid():
    source = raw()
    source['meta']['references'] = [dict(region=region_from_contours([
        [[0,0],[2,0],[2,2],[0,2]],[[2,0],[4,0],[4,2],[2,2]]]))]
    result = build_reference_training_row(source,target_kind=0,real=False)
    delta = torch.tensor([8.,-1.])
    torch.testing.assert_close(result['labels']['angle'][0],delta/delta.norm())
    torch.testing.assert_close(result['labels']['distance'][0],torch.log1p(delta.norm())/torch.log1p(torch.tensor(40.)))
    assert result['labels']['sample_weight'] == 1


def test_changing_training_answer_does_not_change_public_inputs_or_parent():
    source = raw(); before = copy.deepcopy(source)
    first = build_reference_training_row(source,target_kind=0,real=True)
    source['labels']['target_xy'] = torch.tensor([500.,500.])
    second = build_reference_training_row(source,target_kind=0,real=True)
    for section in first['public']:
        for key in first['public'][section]: assert torch.equal(first['public'][section][key],second['public'][section][key])
    assert torch.equal(source['public']['network']['pair_geometry'],before['public']['network']['pair_geometry'])


def test_private_target_cannot_be_hidden_in_public_spatial_input():
    source = raw()
    source['public']['spatial']['target_xy'] = source['labels']['target_xy']
    with pytest.raises(ValueError,match='public spatial'):
        build_reference_training_row(source,target_kind=0,real=True)
