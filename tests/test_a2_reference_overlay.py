import copy
import hashlib

import pytest
import torch

from experiments.a2_waldo_landmark_belief.reference_overlay import (
    make_reference_overlay, apply_reference_overlay, read_reference_overlay, overlay_filename, verify_overlay_population,
)


def example():
    entry = dict(role='train',source_kind='teacher_snapshot',filename='train_1_0.pt',
                 directory='/bound/original',episode=1,step=0,map_name='map',object_id=7,sha256='a'*64)
    network = dict(instruction_tokens=torch.ones(2,768),box_rgb=torch.ones(1,8),
        token_valid=torch.ones(2,dtype=torch.bool),box_valid=torch.ones(1,dtype=torch.bool),
        mention_mask=torch.ones(1,2,dtype=torch.bool),landmark_name_tokens=torch.ones(1,3,768),
        name_valid=torch.ones(1,3,dtype=torch.bool),link_valid=torch.ones(1,dtype=torch.bool),
        pair_geometry=torch.zeros(1,1,14),pair_valid=torch.ones(1,1,dtype=torch.bool))
    meta = dict(references=[dict(map_name='map',object_id=3,name='Road')],reference_surfaces=[dict(z=None)],
                valid_references=1,public_reference_records=1,reference_rebuild_version='public-reference-rebuild-v1',
                reference_rebuild_seconds=.1,extraction_seconds=1.1,candidates=[dict(kind=0)])
    public = dict(network=network,spatial=dict(base_logits=torch.zeros(4)))
    raw = dict(public=copy.deepcopy(public),meta=copy.deepcopy(meta),labels=dict(target_xy=torch.tensor([1.,2.])),
               episode=1,step=0,target_instance=['map',7])
    return raw,entry,public,meta


def test_overlay_contains_only_changed_public_fields_and_parent_binding():
    raw,entry,public,meta = example()
    overlay = make_reference_overlay(public,meta,entry,'b'*64)
    assert 'labels' not in overlay and 'box_rgb' not in overlay['network']
    assert 'target_xy' not in str(overlay.keys())
    restored = apply_reference_overlay(raw,overlay,entry,'b'*64)
    assert torch.equal(restored['public']['network']['box_rgb'],raw['public']['network']['box_rgb'])
    restored['public']['network']['box_rgb'].zero_()
    assert raw['public']['network']['box_rgb'].eq(1).all()
    assert torch.equal(restored['labels']['target_xy'],raw['labels']['target_xy'])


@pytest.mark.parametrize('field,value', [('signature_sha256','c'*64),('kind','foreign')])
def test_foreign_overlay_signature_or_kind_is_rejected(field,value):
    raw,entry,public,meta = example()
    overlay = make_reference_overlay(public,meta,entry,'b'*64);overlay[field]=value
    with pytest.raises(ValueError): apply_reference_overlay(raw,overlay,entry,'b'*64)


def test_parent_role_identity_or_raw_episode_change_is_rejected():
    raw,entry,public,meta = example()
    overlay = make_reference_overlay(public,meta,entry,'b'*64)
    for field,value in [('role','dev'),('sha256','c'*64),('object_id',8),('source_kind','foreign'),('directory','/foreign/source')]:
        changed = dict(entry,**{field:value})
        with pytest.raises(ValueError): apply_reference_overlay(raw,overlay,changed,'b'*64)
    raw['episode'] = 9
    with pytest.raises(ValueError): apply_reference_overlay(raw,overlay,entry,'b'*64)


def test_private_or_unexpected_overlay_fields_are_rejected():
    raw,entry,public,meta = example()
    overlay = make_reference_overlay(public,meta,entry,'b'*64)
    overlay['network']['target_xy'] = torch.zeros(2)
    with pytest.raises(ValueError): apply_reference_overlay(raw,overlay,entry,'b'*64)


def test_bad_axes_or_nonfinite_new_features_are_rejected():
    raw,entry,public,meta = example()
    for key,value in [('pair_geometry',torch.zeros(2,1,14)),('mention_mask',torch.ones(1,3,dtype=torch.bool)),
                      ('landmark_name_tokens',torch.full((1,3,768),float('nan')))]:
        overlay = make_reference_overlay(public,meta,entry,'b'*64)
        overlay['network'][key] = value
        with pytest.raises(ValueError): apply_reference_overlay(raw,overlay,entry,'b'*64)


def test_sealed_role_is_rejected_before_any_overlay_deserialization(monkeypatch,tmp_path):
    _,entry,_,_ = example();entry['role']='calibration_verify'
    def forbidden(*args,**kwargs): raise AssertionError('Must reject before reading tensors')
    monkeypatch.setattr(torch,'load',forbidden)
    with pytest.raises(ValueError):
        read_reference_overlay(tmp_path,dict(entry,overlay_sha256='c'*64),role='train',allowed_targets={('map',7)})


def test_overlay_hash_and_safe_filename_are_checked_before_loading(tmp_path):
    _,entry,public,meta = example()
    overlay = make_reference_overlay(public,meta,entry,'b'*64)
    path = tmp_path/overlay_filename(entry);torch.save(overlay,path)
    record = dict(entry,overlay_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    loaded = read_reference_overlay(tmp_path,record,role='train',allowed_targets={('map',7)})
    assert loaded['signature_sha256'] == 'b'*64
    with pytest.raises(ValueError):
        read_reference_overlay(tmp_path,dict(record,overlay_sha256='d'*64),role='train',allowed_targets={('map',7)})
    assert '/' not in overlay_filename(dict(entry,filename='../malicious.pt'))


def test_complete_population_must_have_exact_parents_and_no_duplicates():
    _,entry,_,_ = example()
    row = dict(entry,overlay_sha256='b'*64)
    verify_overlay_population([row],[entry])
    for rows in ([],[row,row],[dict(row,role='dev')],[dict(row,sha256='d'*64)]):
        with pytest.raises(ValueError): verify_overlay_population(rows,[entry])


def test_dataset_rejects_sealed_roles_before_constructing_source_datasets(monkeypatch):
    from experiments.a2_waldo_landmark_belief import reference_overlay_data as module
    def forbidden(*args,**kwargs): raise AssertionError('No sealed dataset construction')
    monkeypatch.setattr(module,'RelationV3Views',forbidden)
    with pytest.raises(ValueError): module.ReferenceOverlayViews('calibration_verify')
