"""Independent metadata must retain roles and bind the new extraction version."""
import copy
import importlib
import pytest
from tests_confirmation_v3.test_population import fixture, COUNTS


def module():
    return importlib.import_module('experiments.a2_waldo_landmark_belief.reference_v4_confirmation_protocol')


def binding():
    return dict(code_sha256={'reference_v4_train.py':'1'*64},roles_sha256='2'*64,
                feature_manifest_sha256='3'*64,feature_signature_sha256='4'*64,
                feature_build_status_sha256='5'*64,initial_checkpoint_sha256='6'*64)


def metadata(roles=None, context=None):
    r,s,sealed=fixture()
    return module().protocol_metadata(roles or r,s,sealed,binding=binding() if context is None else context,
        source_sha256='7'*64,seal_sha256='8'*64,expected=COUNTS)


def test_fixed_queries_retained_and_new_feature_lineage_changes_metadata():
    value=metadata()
    assert [r['index'] for r in value['fresh_entries']]==[r['index'] for r in fixture()[0]['fresh_calibration']]
    assert value['sealed_entries']==fixture()[2]['targets']
    assert value['target_counts']==dict(fresh=2,fit=1,verify=2,sealed=1)
    assert value['feature_manifest_sha256']=='3'*64
    newer=binding();newer['feature_manifest_sha256']='9'*64
    changed=metadata(context=newer)
    assert changed!=value and changed['fresh_entries']==value['fresh_entries']
    assert value['source_manifest_sha256']=='7'*64
    assert value['sealed_verify_sha256']=='8'*64


@pytest.mark.parametrize('key',list(binding()))
def test_incomplete_or_malformed_binding_rejected_before_population_is_written(key):
    context=binding();del context[key]
    with pytest.raises(ValueError):metadata(context=context)
    context=binding();context[key]={} if key=='code_sha256' else 'not-a-digest'
    with pytest.raises(ValueError):metadata(context=context)


def test_supervised_overlap_and_query_reselection_stay_forbidden():
    roles=copy.deepcopy(fixture()[0])
    roles['target_roles']['train'].append(roles['target_roles']['calibration_fit'][0])
    with pytest.raises(ValueError):metadata(roles)
    roles=copy.deepcopy(fixture()[0]);query=roles['fresh_calibration'][0]
    query['index']=next(i for i in query['episode_indices'] if i!=query['index'])
    with pytest.raises(ValueError):metadata(roles)


def live_fixture(tmp_path,monkeypatch):
    import json
    from experiments.a2_waldo_landmark_belief.calibration_protocol import calibration_fold
    from experiments.a2_waldo_landmark_belief.paths import sha,save
    from experiments.a2_waldo_landmark_belief import reference_v4_preparation
    m=module();base=tmp_path/'runs';old=base/'revision_v3';root=base/'revision_v4'
    fit=[i for i in range(2000) if calibration_fold('m',i)=='fit'][:84]
    verify=[i for i in range(2000) if calibration_fold('m',i)=='verify'][:209]
    targets=fit+verify
    entries=[dict(index=i,map_name='m',object_id=t,description_id=i,split='dev') for i,t in enumerate(targets)]
    roles=dict(target_roles=dict(train=[dict(map_name='m',object_id=3000)],dev=[dict(map_name='m',object_id=3001)],
        calibration_fit=[dict(map_name='m',object_id=t) for t in fit],
        calibration_verify=[dict(map_name='m',object_id=t) for t in verify]),
        fresh_calibration=[dict(map_name='m',object_id=e['object_id'],episode_indices=[e['index']],
            index=e['index'],fold='fit' if e['object_id'] in fit else 'verify') for e in entries[:163]])
    save(dict(entries=entries),base/'source_manifest.json')
    roles['lineage']=dict(original_source_sha256=sha(base/'source_manifest.json'))
    save(roles,old/'roles.json')
    script=tmp_path/'scripts/a2_seal_unscored_verify.py';script.parent.mkdir();script.write_text('# opaque source fixture\n')
    seal=dict(targets=[dict(e,fold='verify') for e in entries[163:]],sealing_code_sha256=sha(script))
    save(seal,old/'sealed_verify130.json')
    save(dict(seal_sha256=sha(old/'sealed_verify130.json')),tmp_path/'reports/a2_execution_20261027/RELATION_V3_VERIFY_SEAL.json')
    context=binding();context['roles_sha256']=sha(old/'roles.json')
    value=m.protocol_metadata(roles,dict(entries=entries),seal,binding=context,
        source_sha256=sha(base/'source_manifest.json'),seal_sha256=sha(old/'sealed_verify130.json'))
    save(value,old/'confirmation_population.json')
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'BASE',base);monkeypatch.setattr(m,'DIRECTORY',root)
    monkeypatch.setattr(reference_v4_preparation,'current_binding',lambda:context)
    # Tiny fixture cannot contain original 1,034 sealed rows; the real opaque
    # verifier remains unchanged and has its own source/byte integrity tests.
    verified=[];monkeypatch.setattr(m,'verify_opaque_seal',lambda s:verified.append(s))
    return m,old,root,verified


def test_prepare_writes_only_new_metadata_and_never_decodes_tensors(tmp_path,monkeypatch):
    import torch
    m,old,root,verified=live_fixture(tmp_path,monkeypatch)
    original={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    def forbidden(*args,**kwargs):raise AssertionError('Metadata preparation must not decode heldout tensors')
    monkeypatch.setattr(torch,'load',forbidden)
    report=m.prepare()
    assert report['fresh_targets']==163 and report['fit_targets']==84 and report['verify_targets']==209
    assert report['sealed_tensor_deserializations']==0 and len(verified)==1
    assert {p.name for p in root.iterdir()}=={'confirmation_population.json','confirmation_protocol.json','confirmation_preparation.json'}
    assert all(p.read_bytes()==data for p,data in original.items())
    written={p:p.read_bytes() for p in root.iterdir()}
    m.prepare()
    assert all(p.read_bytes()==data for p,data in written.items())


def test_prepare_rejects_changed_old_population_before_writing_v4(tmp_path,monkeypatch):
    import json
    m,old,root,_=live_fixture(tmp_path,monkeypatch)
    path=old/'confirmation_population.json';value=json.loads(path.read_text())
    value['fresh_entries'][0]['index']=-1;path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='predeclared'):m.prepare()
    assert not root.exists()
