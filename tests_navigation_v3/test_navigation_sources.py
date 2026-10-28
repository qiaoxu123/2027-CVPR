import json
from types import SimpleNamespace
import pytest
from experiments.a2_waldo_landmark_belief import relation_v3_navigation_sources as sources


def test_formal_population_has_every_original_index_and_same_identity():
    ds=[SimpleNamespace(index=i,map_name='m',object_id=f'o{i}',description_id=0,episode_id=[i]) for i in range(2470)]
    entries=sources.source_entries(ds,'val_seen')
    assert len(entries)==2470 and entries[3]['episode_id']==[3]
    with pytest.raises(ValueError):sources.source_entries(ds[:-1],'val_seen')
    ds[-1].index=0
    with pytest.raises(ValueError):sources.source_entries(ds,'val_seen')


def test_pilot_entry_loader_revalidates_fixed_metadata_before_return(tmp_path,monkeypatch):
    rows=[dict(index=i) for i in range(64)]
    (tmp_path/'navigation_pilot_population.json').write_text(json.dumps(dict(entries=rows)))
    called=[];monkeypatch.setattr(sources,'prepare_pilot',lambda root:called.append(root))
    assert sources.source_entries([], 'pilot',root=tmp_path)==rows
    assert called==[tmp_path]


def test_verification_rejects_trace_missing_from_combined_file(tmp_path):
    rows=[dict(index=1),dict(index=2)]
    path=tmp_path/'predictions.jsonl';path.write_text(json.dumps(rows[0])+'\n')
    with pytest.raises(ValueError):sources.check_combined(path,rows)
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows));sources.check_combined(path,rows)
    path.write_text(path.read_text()+json.dumps(rows[1])+'\n')
    with pytest.raises(ValueError):sources.check_combined(path,rows)
