"""Real small tensors exercise V4 FIT/VERIFY barriers without live heldout data."""
import copy
import importlib
import json
import pytest
import torch
from experiments.a2_waldo_landmark_belief.paths import sha,save
from tests_confirmation_v3.test_packet import row_and_model
from tests_confirmation_v4.test_collection import setup


def module():
    return importlib.import_module('experiments.a2_waldo_landmark_belief.reference_v4_calibration_data')


def source_fixture(tmp_path,monkeypatch):
    root,state,checkpoint=setup(tmp_path,monkeypatch)
    from experiments.a2_waldo_landmark_belief.reference_v4_confirmation import collection_signature
    row,model=row_and_model();state.update(model_config=model.config,model=model.state_dict())
    directory=root/'confirmation';directory.mkdir()
    save(collection_signature(root,state,checkpoint),directory/'signature.json');signature=sha(directory/'signature.json')
    population=json.loads((root/'confirmation_population.json').read_text());entries=[];episodes=[]
    for query in population['fresh_entries']:
        item=copy.deepcopy(row);item.update(signature_sha256=signature,episode=query['index'],step=0,
            target_instance=[query['map_name'],query['object_id']],fold=query['fold'])
        name=f"view_{query['index']}_0.pt";torch.save(item,directory/name)
        entry=dict(filename=name,sha256=sha(directory/name),episode=query['index'],step=0,fold=query['fold']);entries.append(entry)
        complete=directory/f"episode_{query['index']}_complete.json"
        save(dict(entry=query,signature_sha256=signature,rows=[entry]),complete)
        save(dict(complete_sha256=sha(complete),signature_sha256=signature,index=query['index']),complete.with_suffix('.sha.json'))
        episodes.append(dict(index=query['index'],complete_file=complete.name,sha256=sha(complete),sidecar_sha256=sha(complete.with_suffix('.sha.json'))))
    save(dict(status='v4_fresh163_independent_A0_confirmation_complete',target_groups=163,
              signature_sha256=signature,episodes=episodes,entries=entries),directory/'manifest.json')
    m=module();monkeypatch.setattr(m,'BASE',root.parent);monkeypatch.setattr(m,'checked_state',lambda path:state)
    monkeypatch.setattr(m,'verify_cache_origin',lambda path:None)
    return m,root,state


def test_fresh_dataset_checks_complete_v4_lineage_without_decoding_verify(tmp_path,monkeypatch):
    m,root,_=source_fixture(tmp_path,monkeypatch)
    original=torch.load;seen=[]
    def guarded(path,*args,**kwargs):
        seen.append(str(path));return original(path,*args,**kwargs)
    monkeypatch.setattr(torch,'load',guarded)
    source=m.FreshConfirmation(root)
    assert len(source.groups('fit'))==84 and len(source.groups('verify'))==79
    assert seen==[]
    with pytest.raises(FileNotFoundError):source.packets('verify')
    assert seen==[] and not (root/'calibration/packets_fresh_verify').exists()


def test_complete_fit_packets_keep_saved_model_scores_and_feature_signature(tmp_path,monkeypatch):
    m,root,state=source_fixture(tmp_path,monkeypatch);source=m.FreshConfirmation(root)
    save(dict({key:state[key] for key in ('code_sha256','roles_sha256','feature_manifest_sha256',
        'feature_signature_sha256','feature_build_status_sha256','initial_checkpoint_sha256')},
        checkpoint_sha256=source.checkpoint_sha256),
        root/'calibration/analysis_signature.json')
    packets=source.packets('fit')
    assert len(packets)==84 and all(p['fold']=='fit' for p in packets)
    assert all('V4' in p['inference_authority'] for p in packets)
    directory=root/'calibration/packets_fresh_fit'
    signature=json.loads((directory/'signature.json').read_text())
    assert signature['feature_manifest_sha256']==state['feature_manifest_sha256']
    m.validate_fit_cache(root,binding=state)
    first=directory/'view_0_0.pt';first.write_bytes(b'corrupt')
    with pytest.raises(ValueError):m.validate_fit_cache(root,binding=state)


@pytest.mark.parametrize('key',['feature_manifest_sha256','feature_signature_sha256','feature_build_status_sha256'])
def test_dataset_refuses_rebound_features_before_tensor_loading(tmp_path,monkeypatch,key):
    m,root,state=source_fixture(tmp_path,monkeypatch);state[key]='9'*64
    monkeypatch.setattr(torch,'load',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Premature tensor load')))
    with pytest.raises(ValueError):m.FreshConfirmation(root)


def test_no_fit_candidate_never_opens_heldout_tensor_or_other_files(tmp_path):
    save(dict(selected_parameters=None),tmp_path/'calibration/fit_decision.json')
    with pytest.raises(ValueError,match='No FIT'):
        module().load_frozen_choice(tmp_path,'checkpoint','roles',binding={})


def test_fit_cache_rejects_other_features_even_with_updated_manifest_hash(tmp_path,monkeypatch):
    m,root,state=source_fixture(tmp_path,monkeypatch);source=m.FreshConfirmation(root)
    save(dict({key:state[key] for key in ('code_sha256','roles_sha256','feature_manifest_sha256',
        'feature_signature_sha256','feature_build_status_sha256','initial_checkpoint_sha256')},
        checkpoint_sha256=source.checkpoint_sha256),
        root/'calibration/analysis_signature.json')
    source.packets('fit');path=root/'calibration/packets_fresh_fit/signature.json'
    signature=json.loads(path.read_text());signature['feature_manifest_sha256']='9'*64;save(signature,path)
    with pytest.raises(ValueError,match='feature binding'):m.validate_fit_cache(root,binding=state)


def test_missing_frozen_fit_choice_blocks_verify_before_first_packet_load(tmp_path,monkeypatch):
    m,root,_=source_fixture(tmp_path,monkeypatch);source=m.FreshConfirmation(root)
    save(dict(selected_parameters=None),root/'calibration/fit_decision.json')
    monkeypatch.setattr(torch,'load',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Premature VERIFY load')))
    with pytest.raises(ValueError,match='No FIT'):source.packets('verify')
    assert not (root/'calibration/packets_fresh_verify').exists()


def test_fit_cache_rejects_another_trained_model_even_if_all_commit_hashes_agree(tmp_path,monkeypatch):
    m,root,state=source_fixture(tmp_path,monkeypatch);source=m.FreshConfirmation(root)
    keys=('code_sha256','roles_sha256','feature_manifest_sha256','feature_signature_sha256',
          'feature_build_status_sha256','initial_checkpoint_sha256')
    save(dict({key:state[key] for key in keys},checkpoint_sha256=source.checkpoint_sha256),
         root/'calibration/analysis_signature.json')
    source.packets('fit');m.validate_fit_cache(root,binding=state)
    directory=root/'calibration/packets_fresh_fit';path=directory/'signature.json'
    signature=json.loads(path.read_text());signature['checkpoint_sha256']='9'*64;save(signature,path)
    digest=sha(path);manifest=json.loads((directory/'manifest.json').read_text())
    for entry in manifest['entries']:
        sidecar=(directory/entry['filename']).with_suffix('.sha.json')
        value=json.loads(sidecar.read_text());value['signature_sha256']=digest;save(value,sidecar)
        entry['sidecar_sha256']=sha(sidecar)
    manifest['signature_sha256']=digest;save(manifest,directory/'manifest.json')
    with pytest.raises(ValueError,match='checkpoint'):
        m.validate_fit_cache(root,binding=state)
