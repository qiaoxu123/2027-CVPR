import pytest
import torch
from experiments.a2_waldo_landmark_belief.relation_v3_sealed_replay import read_sealed_original,check_replay_population,infer_row


def test_sealed_read_requires_frozen_fit_before_any_tensor_read(monkeypatch,tmp_path):
    monkeypatch.setattr(torch,'load',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Premature sealed read')))
    with pytest.raises(ValueError):read_sealed_original(tmp_path,{}, {},'signature',authorized=False)


def test_replay_population_rejects_missing_duplicate_or_rebound_rows():
    expected=[dict(filename='view_1_0.pt',episode=1,step=0,raw_sha256='raw1'),dict(filename='view_2_0.pt',episode=2,step=0,raw_sha256='raw2')]
    rows=[dict(filename=e['filename'],episode=e['episode'],step=e['step'],original_raw_sha256=e['raw_sha256']) for e in expected]
    check_replay_population(rows,expected,{1,2})
    with pytest.raises(ValueError):check_replay_population(rows[:-1],expected,{1,2})
    with pytest.raises(ValueError):check_replay_population(rows+[rows[0]],expected,{1,2})
    rows[0]['original_raw_sha256']='different'
    with pytest.raises(ValueError):check_replay_population(rows,expected,{1,2})


def test_private_target_cannot_change_replayed_public_predictions():
    import copy
    from tests_confirmation_v3.test_packet import row_and_model
    row,model=row_and_model();first=infer_row(row,model,'sig','raw',device='cpu')
    changed=copy.deepcopy(row);changed['labels']['target_xy']=torch.tensor([999.,-999.])
    second=infer_row(changed,model,'sig','raw',device='cpu')
    assert not torch.equal(first['labels']['target_xy'],second['labels']['target_xy'])
    for name in first['risk']:assert torch.equal(first['risk'][name],second['risk'][name])
    for name in first['prediction']:assert torch.equal(first['prediction'][name],second['prediction'][name])
