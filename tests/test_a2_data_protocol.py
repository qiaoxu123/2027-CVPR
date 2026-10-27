from types import SimpleNamespace
import pytest
from experiments.a2_waldo_landmark_belief.data_protocol import split_target_groups,public_reference_records,validate_source_manifest


def test_target_instances_do_not_cross_train_dev_calibration():
    rows=[SimpleNamespace(map_name='m',object_id=i//3)for i in range(600)]
    split=split_target_groups(rows,seed=0)
    for i in range(0,len(rows),3):assert len({split[i+j]for j in range(3)})==1
    assert set(split)=={'train','dev','calibration'}
    reversed_split=split_target_groups(rows[::-1],seed=0)
    assert split==list(reversed(reversed_split))


def test_public_reference_linking_rejects_fuzzy_wrong_identity_and_merges_same_id():
    refs=[dict(map_name='m',object_id=1,name='Church',query_name='Church',contour=[[0,0],[2,0],[2,2]]),dict(map_name='m',object_id=1,name='Church',query_name='church',contour=[[0,0],[2,0],[2,2]]),dict(map_name='m',object_id=2,name='Library',query_name='Librery',contour=[[3,3],[5,3],[5,5]])]
    records=public_reference_records(refs)
    assert len(records)==2 and records[0]['linked']
    assert not records[1]['linked']


def test_distinct_public_ids_with_same_mention_are_ambiguous():
    refs=[dict(map_name='m',object_id=i,name='Bank',query_name='Bank',contour=[[i,0],[i+1,0],[i+1,1]])for i in (1,3)]
    assert not any(r['linked']for r in public_reference_records(refs))


def test_changed_annotation_and_reordered_episode_identity_fail_before_extraction(tmp_path):
    import hashlib
    path=tmp_path/'annotation.json';path.write_text('original')
    row=SimpleNamespace(map_name='m',object_id=1,description_id=2)
    class Dataset:
        annotation_path=path
        def __getitem__(self,index):return row
        def __len__(self):return 1
    source=dict(annotation_sha256=hashlib.sha256(b'original').hexdigest(),entries=[dict(index=0,map_name='m',object_id=1,description_id=2,
                split=split_target_groups([row])[0])])
    validate_source_manifest(source,Dataset())
    path.write_text('modified')
    with pytest.raises(ValueError,match='annotation'):validate_source_manifest(source,Dataset())
    path.write_text('original');source['entries'][0]['object_id']=3
    with pytest.raises(ValueError,match='identity'):validate_source_manifest(source,Dataset())
