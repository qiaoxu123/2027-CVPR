import copy
import numpy as np
import pytest
import torch

from experiments.a2_waldo_landmark_belief.public_road_references import resolve_road_references
from experiments.a2_waldo_landmark_belief.reference_rebuild import rebuild_reference_pack


def fixture():
    contour = [[0,0],[1,0],[1,1],[0,1]]
    candidate = dict(poly=contour, confidence=.9, surface=dict(z=12.,coverage=.8), kind=0)
    pack = dict(network=dict(instruction_tokens=torch.ones(2,768),token_valid=torch.ones(2,dtype=torch.bool),
        box_rgb=torch.ones(1,8),box_valid=torch.ones(1,dtype=torch.bool),offset_limit=torch.zeros(1,2),
        mention_mask=torch.zeros(1,2,dtype=torch.bool),landmark_name_tokens=torch.zeros(1,1,768),
        name_valid=torch.zeros(1,1,dtype=torch.bool),link_valid=torch.zeros(1,dtype=torch.bool),
        pair_geometry=torch.zeros(1,1,14),pair_valid=torch.zeros(1,1,dtype=torch.bool)),
        spatial=dict(base_logits=torch.arange(4.),grid_world=torch.zeros(4,2)))
    meta = dict(candidates=[candidate],references=[],reference_surfaces=[],metres_per_pixel=.5,
                extraction_seconds=1.,valid_references=0)
    objects = [dict(id=i,name='Main Road',object_type=kind,contour=np.add(contour,[i-1,0]).tolist())
               for i,kind in [(1,'TrafficRoad'),(2,'Footpath')]]
    return pack,meta,resolve_road_references('map',['Main Road'],objects)


def encode_names(names):
    assert names == ['Main Road; Footpath; TrafficRoad']
    return torch.full((len(names),3,768),7.), torch.ones(len(names),3,dtype=torch.bool)


def call(pack,meta,refs,encoder=encode_names):
    return rebuild_reference_pack(pack,meta,instruction='Main Road',offsets=[[0,4],[5,9]],
                                  references=refs,encode_names=encoder)


def test_group_rebuild_restores_features_and_reuses_images_without_parent_mutation():
    pack,meta,refs = fixture()
    before = copy.deepcopy(pack)
    new,info = call(pack,meta,refs)
    assert info['valid_references'] == 1 and new['network']['link_valid'].tolist() == [True]
    assert new['network']['mention_mask'].tolist() == [[True,True]]
    assert new['network']['landmark_name_tokens'].shape == (1,3,768)
    assert new['network']['landmark_name_tokens'].eq(7).all()
    assert new['network']['pair_valid'].all()
    assert new['network']['pair_geometry'][0,0,10:12].tolist() == [0,0]
    assert new['network']['pair_geometry'][0,0,12] == pytest.approx(.5/40)
    assert new['network']['pair_geometry'][0,0,13] == pytest.approx(.8)
    assert torch.equal(new['network']['box_rgb'],pack['network']['box_rgb'])
    new['network']['box_rgb'].zero_()
    new['spatial']['base_logits'].zero_()
    for section in pack:
        for key in pack[section]: assert torch.equal(pack[section][key],before[section][key])
    assert meta['references'] == [] and info['reference_surfaces'][0]['z'] is None


def test_no_usable_reference_keeps_padding_and_never_encodes_guessed_names():
    pack,meta,refs = fixture()
    refs[0]['linked'] = False
    def forbidden(_): raise AssertionError('No unconfirmed names may be encoded')
    new,info = call(pack,meta,refs,forbidden)
    assert info['valid_references'] == 0
    assert not new['network']['pair_valid'].any() and not new['network']['link_valid'].any()
    assert new['network']['landmark_name_tokens'].shape == (1,1,768)


def test_existing_singleton_reuses_real_surface_and_new_singleton_is_explicitly_missing():
    pack,meta,_ = fixture()
    ref = dict(map_name='map',object_id=9,name='Main Road',aliases=['Main Road'],linked=True,
               contour=[[0,0],[1,0],[1,1],[0,1]])
    encoder = lambda names: (torch.zeros(len(names),1,768),torch.ones(len(names),1,dtype=torch.bool))
    missing,missing_meta = call(pack,meta,[ref],encoder)
    assert missing_meta['references'][0]['height_policy'] == 'missing_newly_available_reference'
    assert missing['network']['pair_geometry'][0,0,11] == 0
    meta['references'] = [copy.deepcopy(ref)]
    meta['reference_surfaces'] = [dict(z=2.,valid=True,coverage=.9)]
    existing,existing_meta = call(pack,meta,[ref],encoder)
    assert existing['network']['pair_geometry'][0,0,10] == pytest.approx(.5)
    assert existing['network']['pair_geometry'][0,0,11] == 1
    assert existing_meta['reference_surfaces'][0] == meta['reference_surfaces'][0]


def test_offset_alignment_and_encoder_shape_fail_before_saving_any_features():
    pack,meta,refs = fixture()
    pack['network']['instruction_tokens'] = torch.ones(3,768)
    with pytest.raises(ValueError,match='token'): call(pack,meta,refs)
    pack,meta,refs = fixture()
    with pytest.raises(ValueError,match='name feature'):
        call(pack,meta,refs,lambda names: (torch.zeros(2,3,768),torch.ones(2,3,dtype=torch.bool)))


def test_empty_candidate_padding_does_not_create_real_pairs():
    pack,meta,refs = fixture()
    meta['candidates'] = []
    pack['network']['box_valid'].zero_()
    new,info = call(pack,meta,refs)
    assert info['valid_references'] == 1 and not new['network']['pair_valid'].any()
    assert new['network']['pair_geometry'].shape == (1,1,14)


def test_private_fields_are_rejected_and_masked_tokens_do_not_restore_mentions():
    pack,meta,refs = fixture()
    pack['network']['target_xy'] = torch.tensor([1.,2.])
    with pytest.raises(ValueError,match='public network'): call(pack,meta,refs)
    del pack['network']['target_xy']
    pack['spatial']['target_xy'] = torch.tensor([1.,2.])
    with pytest.raises(ValueError,match='public spatial'): call(pack,meta,refs)
    del pack['spatial']['target_xy']
    pack['network']['token_valid'].zero_()
    def forbidden(_): raise AssertionError('Masked instruction must not authorize names')
    new,info = call(pack,meta,refs,forbidden)
    assert info['valid_references'] == 0 and not new['network']['link_valid'].any()
