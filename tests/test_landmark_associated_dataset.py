import pytest
import json
from scripts.build_landmark_associated_train import build, decision, classification_index, task_key, sha


def test_keep_all_confirmed_association_classes_and_remove_only_requested_classes():
    for name in ('direct_geometry','direct_identity','indirect_geometry'):
        assert decision({'category':name})
    for name in ('uncertain','no_confirmed_anchor'):
        assert not decision({'category':name})
    with pytest.raises(ValueError):
        decision({'category':'new_unknown_label'})


def test_join_is_map_and_exact_instruction_not_shifted_processed_index():
    assert task_key(dict(area='cambridge',block=1,descriptions=['Red car.'])) == ('cambridge_block_1','Red car.')
    a=dict(map_name='cambridge_block_1',instruction='Red car.',category='direct_geometry',target_category='Car')
    b=dict(a, map_name='cambridge_block_2',category='uncertain')
    assert len(classification_index([a,b])) == 2
    with pytest.raises(ValueError):
        classification_index([a,dict(a,category='uncertain')])


def test_materialization_preserves_rows_and_existing_partitions(tmp_path):
    original = tmp_path/'original'; original.mkdir()
    processed = tmp_path/'processed'; processed.mkdir()
    rows = [dict(area='map',block=i,descriptions=['instruction '+str(i)],object_ids=[7],
                 trajectory=[[i,2,3]],target_positions=[[10,20]]) for i in range(4)]
    source = original/'citynav_train_seen.json'; source.write_text(json.dumps(rows))
    # The processed source has a different ordering: matching must use text/map.
    proc = processed/'citynav_train_seen.json'; proc.write_text(json.dumps(rows[::-1]))
    names = ['direct_geometry','uncertain','indirect_geometry','no_confirmed_anchor']
    records = [dict(episode=i,map_name='map_block_'+str(i),instruction=r['descriptions'][0],
                    category=names[i],target_category='Car') for i,r in enumerate(rows)]
    rec = tmp_path/'records.json'; rec.write_text(json.dumps(records))
    summary = tmp_path/'summary.json'; summary.write_text(json.dumps(dict(n=4,provenance=dict(source_sha256=sha(source)))))
    maps = tmp_path/'maps.json'; maps.write_text(json.dumps(dict(map_assignment={
        'map_block_0':'fit','map_block_1':'dev','map_block_2':'confirm','map_block_3':'dev'})))
    out = tmp_path/'output'
    manifest = build(source,proc,rec,summary,maps,out)
    assert manifest['official_retained'] == 2
    assert json.loads((out/'official_train_associated.json').read_text()) == [rows[0],rows[2]]
    assert json.loads((out/'processed_all.json').read_text()) == [rows[2],rows[0]]
    assert json.loads((out/'processed_fit.json').read_text()) == [rows[0]]
    assert json.loads((out/'processed_dev.json').read_text()) == []
    assert json.loads((out/'processed_confirm.json').read_text()) == [rows[2]]
    assert len((out/'removed_records.jsonl').read_text().splitlines()) == 2
    with pytest.raises(FileExistsError):
        build(source,proc,rec,summary,maps,out)
