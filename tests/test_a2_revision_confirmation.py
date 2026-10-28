import pytest
from experiments.a2_waldo_landmark_belief.revision_confirmation import confirmation_entries,immutable_json


def test_confirmation_query_selection_is_target_disjoint_and_never_reads_outcomes():
    source=dict(entries=[dict(index=i,map_name='m',object_id=oid,description_id=i,split='dev') for i,oid in enumerate([1,1,2,2])])
    reserve=dict(targets=[dict(map_name='m',object_id=oid,episode_indices=[i for i in range(4) if source['entries'][i]['object_id']==oid]) for oid in [1,2]])
    entries=confirmation_entries(reserve,source)
    assert len(entries)==2 and {e['object_id'] for e in entries}=={1,2}
    assert entries==confirmation_entries(reserve,dict(entries=list(reversed(source['entries']))))
    assert all(e['fold'] in ('fit','verify') for e in entries)
    source['entries'][0]['object_id']=99
    with pytest.raises(ValueError):confirmation_entries(reserve,source)


def test_confirmation_rejects_training_targets_and_manifest_changes(tmp_path):
    source=dict(entries=[dict(index=0,map_name='m',object_id=1,description_id=0,split='train')])
    reserve=dict(targets=[dict(map_name='m',object_id=1,episode_indices=[0])])
    with pytest.raises(ValueError):confirmation_entries(reserve,source)
    p=tmp_path/'manifest.json';immutable_json(dict(a=1),p);immutable_json(dict(a=1),p)
    with pytest.raises(ValueError):immutable_json(dict(a=2),p)
