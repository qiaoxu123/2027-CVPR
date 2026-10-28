"""The evaluator receives every original episode; partial or repeated sets fail."""
import importlib
from types import SimpleNamespace
import pytest


def module():
    return importlib.import_module('experiments.a2_waldo_landmark_belief.reference_v4_navigation_population')


@pytest.mark.parametrize('scope,count',[('val_seen',2470),('val_unseen',2697)])
def test_full_population_preserves_each_episode_identity_and_rejects_subsets(scope,count):
    dataset=[SimpleNamespace(index=i,map_name='m',object_id=i,description_id=0,episode_id=(i,)) for i in range(count)]
    entries=module().source_entries(dataset,scope)
    assert len(entries)==count and entries[-1]['index']==count-1 and entries[3]['episode_id']==[3]
    assert not any('target_position' in e for e in entries)
    with pytest.raises(ValueError):module().source_entries(dataset[:128],scope)
    dataset[-1].index=0
    with pytest.raises(ValueError):module().source_entries(dataset,scope)


def test_unsupported_scope_cannot_be_treated_as_complete_validation():
    with pytest.raises(ValueError):module().source_entries([],'seen128')


def test_real_pilot_loader_derives_missing_episode_id_without_rewriting_fixed_queries(tmp_path,monkeypatch):
    import json
    from experiments.a2_waldo_landmark_belief.paths import save
    m=module();rows=[dict(index=i,map_name='m',object_id=i,description_id=0,fold='verify') for i in range(64)]
    save(dict(entries=rows),tmp_path/'navigation_pilot_population.json')
    original=(tmp_path/'navigation_pilot_population.json').read_bytes()
    dataset=[SimpleNamespace(index=i,map_name='m',object_id=i,description_id=0,episode_id=('m',i,0)) for i in range(64)]
    monkeypatch.setattr(m,'prepare_pilot',lambda root:None)
    entries=m.source_entries(dataset,'pilot',root=tmp_path)
    assert entries[3]['episode_id']==['m',3,0]
    assert entries[3]['fold']=='verify' and 'episode_id' not in rows[3]
    assert (tmp_path/'navigation_pilot_population.json').read_bytes()==original
    dataset[3].object_id=999
    with pytest.raises(ValueError):m.source_entries(dataset,'pilot',root=tmp_path)
