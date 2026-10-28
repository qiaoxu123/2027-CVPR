import pytest
import torch
from experiments.a2_waldo_landmark_belief.relation_v3_data import build_training_row,read_training_source


def raw():
    return dict(target_instance=['m',7],episode=1,step=0,signature_sha256='sig',labels=dict(target_xy=torch.tensor([0.,0.])),
        private=dict(association=[dict(object_id=999,trusted=True)]),
        meta=dict(candidates=[dict(kind=0),dict(kind=1)],references=[dict(contour=[[-1,-1],[1,-1],[1,1],[-1,1]])]),
        public=dict(network=dict(box_rgb=torch.ones(2,6),box_valid=torch.ones(2,dtype=torch.bool),pair_valid=torch.ones(2,1,dtype=torch.bool),
            instruction_tokens=torch.zeros(2,8),landmark_name_tokens=torch.zeros(1,2,8),pair_geometry=torch.zeros(2,1,14),offset_limit=torch.zeros(2,2),
            link_valid=torch.ones(1,dtype=torch.bool),mention_mask=torch.ones(1,2,dtype=torch.bool),token_valid=torch.ones(2,dtype=torch.bool),name_valid=torch.ones(1,2,dtype=torch.bool)),
            spatial=dict(base_logits=torch.zeros(2),candidate_xy=torch.zeros(2,2),grid_world=torch.tensor([[0.,0.],[100.,0.]]),candidate_support=torch.ones(2,2,dtype=torch.bool))))


def test_training_labels_use_own_target_and_public_geometry_not_private_identity_associations():
    r=build_training_row(raw(),target_kind=0,real=True)
    assert r['labels']['localization_utility'].tolist()==[True,False]
    assert not r['labels']['identity_label_mask'].any()
    assert r['labels']['sample_weight']==8 and r['labels']['source_is_real']
    assert 'private' not in r and 'target_xy' not in r['public']['network']


def test_calibration_reference_is_rejected_before_any_tensor_read(monkeypatch):
    def forbidden(*a,**k):raise AssertionError('Sealed tensor must never be deserialized')
    monkeypatch.setattr(torch,'load',forbidden)
    reference=dict(role='calibration_verify',map_name='m',object_id=7)
    with pytest.raises(ValueError):read_training_source(reference,'train',{('m',7)},'sig')
    reference['role']='train';reference['object_id']=999
    with pytest.raises(ValueError):read_training_source(reference,'train',{('m',7)},'sig')


def test_duplicate_observation_rejects_canonical_directory_aliases(tmp_path):
    from experiments.a2_waldo_landmark_belief.relation_v3_data import observation_keys
    rows=[dict(directory=str(tmp_path),filename='x.pt'),dict(directory=str(tmp_path)+'/.',filename='x.pt')]
    with pytest.raises(ValueError,match='Duplicate'):observation_keys(rows)
