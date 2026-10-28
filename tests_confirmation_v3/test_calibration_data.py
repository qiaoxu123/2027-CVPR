import json
import pytest
import torch
from experiments.a2_waldo_landmark_belief.paths import sha
from experiments.a2_waldo_landmark_belief.calibration_protocol import calibration_fold
from experiments.a2_waldo_landmark_belief.relation_v3_calibration_data import read_fresh_row,load_frozen_choice,validate_fit_cache


def create_fit_cache(root):
    directory=root/'calibration/packets_fresh_fit';directory.mkdir(parents=True)
    (root/'confirmation').mkdir();source=[]
    for i in range(84):source.append(dict(filename=f'view_{i}_0.pt',sha256=f'raw{i}',episode=i,step=0,fold='fit'))
    manifest=root/'confirmation/manifest.json';manifest.write_text(json.dumps(dict(entries=source)))
    (root/'calibration/analysis_signature.json').write_text('{}')
    signature=dict(fold='fit',source_manifest_sha256=sha(manifest),analysis_signature_sha256=sha(root/'calibration/analysis_signature.json'))
    (directory/'signature.json').write_text(json.dumps(signature));digest=sha(directory/'signature.json');entries=[]
    for e in source:
        path=directory/e['filename'];path.write_text('opaque unit-test packet')
        sidecar=path.with_suffix('.sha.json');sidecar.write_text(json.dumps(dict(packet_sha256=sha(path),signature_sha256=digest,source_row_sha256=e['sha256'])))
        entries.append(dict(filename=e['filename'],sha256=sha(path),sidecar_sha256=sha(sidecar),source_row_sha256=e['sha256']))
    (directory/'manifest.json').write_text(json.dumps(dict(signature_sha256=digest,fold='fit',target_groups=84,entries=entries)))
    return directory


def test_cross_fold_row_is_rejected_before_any_tensor_deserialization(monkeypatch,tmp_path):
    monkeypatch.setattr(torch,'load',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Premature heldout tensor read')))
    query=dict(index=1,map_name='m',object_id=1,fold='verify');entry=dict(episode=1,step=0,fold='verify')
    with pytest.raises(ValueError):read_fresh_row(tmp_path,entry,query,'sig','fit')
    with pytest.raises(ValueError):read_fresh_row(tmp_path,entry,query,'sig','verify')


def test_fit_raw_identity_must_match_fixed_query_even_if_file_hash_matches(tmp_path):
    target=next(i for i in range(100) if calibration_fold('m',i)=='fit');query=dict(index=1,map_name='m',object_id=target,fold='fit')
    row=dict(signature_sha256='sig',episode=1,step=0,fold='fit',target_instance=['m',target])
    path=tmp_path/'view_1_0.pt';torch.save(row,path);entry=dict(filename=path.name,sha256=sha(path),episode=1,step=0,fold='fit')
    assert read_fresh_row(tmp_path,entry,query,'sig','fit')['target_instance']==['m',target]
    row['target_instance']=['m',999];torch.save(row,path);entry['sha256']=sha(path)
    with pytest.raises(ValueError):read_fresh_row(tmp_path,entry,query,'sig','fit')


def test_no_selected_fit_choice_fails_without_reading_any_other_file(tmp_path):
    (tmp_path/'calibration').mkdir();(tmp_path/'calibration/fit_decision.json').write_text(json.dumps(dict(selected_parameters=None)))
    with pytest.raises(ValueError,match='No FIT'):load_frozen_choice(tmp_path,'checkpoint','roles')


def test_frozen_fit_cache_checks_each_actual_packet_and_sidecar(tmp_path):
    directory=create_fit_cache(tmp_path);validate_fit_cache(tmp_path)
    path=directory/'view_3_0.pt';old=path.read_bytes();path.write_bytes(b'corrupted')
    with pytest.raises(ValueError):validate_fit_cache(tmp_path)
    path.write_bytes(old);path.with_suffix('.sha.json').write_text('{}')
    with pytest.raises(ValueError):validate_fit_cache(tmp_path)
