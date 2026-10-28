"""Export stops before data access until admitted, and preserves role lineage."""
import importlib
from types import SimpleNamespace
import json

import pytest
import torch

from diagnostics_tests.evidence_quality.test_packets import population
from diagnostics_tests.evidence_quality.test_data import example


def api():
    return importlib.import_module('diagnostics.evidence_quality.export')


def datasets():
    train, _ = population('train'); dev, _ = population('dev')
    for row in dev: row['map_name']='heldout'
    return dict(train=SimpleNamespace(entries=train),dev=SimpleNamespace(entries=dev)), dict(rows=train+dev)


def test_actual_wait_does_not_construct_dataset_or_touch_output(tmp_path,monkeypatch):
    module=api();monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(module,'admit',lambda root:None)
    assert module.export_packets(tmp_path/'root',tmp_path/'output') is None
    assert not (tmp_path/'output').exists()


def test_selects_exact_same_ordered_real_role_sources():
    data, roles=datasets(); selected=api().select_population(data,roles)
    assert len(selected['train'])==2025 and len(selected['dev'])==1029
    assert selected['train'][0]['filename']=='view_0.pt'


@pytest.mark.parametrize('change',['target_overlap','reordered','foreign_role','source_changed'])
def test_foreign_or_overlapping_target_population_rejected(change):
    data, roles=datasets()
    # Independent role metadata must not change along with the dataset mutation.
    roles=dict(rows=[dict(e) for e in roles['rows']])
    if change=='target_overlap':
        for e in data['dev'].entries:e['map_name']='map'
        for e in roles['rows']:
            if e['role']=='dev':e['map_name']='map'
    if change=='reordered':data['train'].entries.reverse()
    if change=='foreign_role':data['train'].entries[0]['role']='calibration'
    if change=='source_changed':data['train'].entries[0]['sha256']='c'*64
    with pytest.raises(ValueError):api().select_population(data,roles)


def test_full_population_actual_cpu_forward_export_and_cache_reuse(tmp_path,monkeypatch):
    module=api();monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    metadata,roles=datasets(); model,public=example()
    root=tmp_path/'base/revision_v4';root.mkdir(parents=True)
    role_path=root.parent/'revision_v3/roles.json';role_path.parent.mkdir()
    role_path.write_text(json.dumps(roles))
    checkpoint=root/'selected.pt'
    torch.save(dict(model_config=model.config,model=model.state_dict()),checkpoint)
    admission=dict(checkpoint=str(checkpoint),checkpoint_sha256=module.sha(checkpoint),
                   roles_sha256=module.sha(role_path),feature_manifest_sha256='f'*64)

    class CachedSource:
        manifest_sha256='f'*64
        forbid=False
        def __init__(self,entries):self.entries=entries
        def __getitem__(self,index):
            assert not self.forbid, 'Reusing committed cache must not rerun inference'
            return dict(public=public,labels=dict(target_xy=torch.tensor([0.,0.])))

    sources={role:CachedSource(data.entries) for role,data in metadata.items()}
    # Replace unavailable original data/complete-job admission, retain actual
    # CPU network, feature extraction, outcome attachment and atomic storage.
    monkeypatch.setattr(module,'admit',lambda _:admission)
    monkeypatch.setattr(module,'ReferenceOverlayViews',lambda role:sources[role])
    output=tmp_path/'export'
    result=module.export_packets(root,output)
    assert {k:v['views'] for k,v in result['manifests'].items()}==dict(train=2025,dev=1029)
    for source in sources.values():source.forbid=True
    reused=module.export_packets(root,output)
    assert reused==result
    original_load=module.load_packets
    def mutate_checkpoint_after_last_cached_read(*args,**kwargs):
        loaded=original_load(*args,**kwargs)
        if kwargs['role']=='dev':checkpoint.write_bytes(checkpoint.read_bytes()+b'changed')
        return loaded
    monkeypatch.setattr(module,'load_packets',mutate_checkpoint_after_last_cached_read)
    with pytest.raises(ValueError):module.export_packets(root,output)
