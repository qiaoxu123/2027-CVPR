"""HETT official Refined vs local CityNav revised provenance, no name guessing."""
import json
import pytest

from scripts.audit_hett_refined_data import (
    audit, compare_assets, episodes, processed_keys,
)


def _episodes(path,rows):
    path.write_text(json.dumps(rows))
    return path


def _episode(obj,ann,description):
    return dict(area="cambridge",block=7,
                object_ids=[obj],ann_ids=[ann],
                descriptions=[description],target_positions=[[10,20,100]])


def test_identical_official_local_means_verified_same_episode_content(tmp_path):
    current=_episodes(tmp_path/"current.json",[_episode(1,0,"House north of One Stop")])
    official=_episodes(tmp_path/"official.json",[_episode(1,0,"House north of One Stop")])
    result=audit(current,official)
    assert result["val_seen_exact_official_refined"] is True
    assert result["val_seen_semantically_equivalent"] is True
    assert result["val_seen"]["only_current"]==0


def test_revised_label_not_equal_official_changed_description_or_deleted(tmp_path):
    current=_episodes(tmp_path/"current.json",[
        _episode(1,0,"House beside X"),
        _episode(2,0,"Blue car"),
    ])
    official=_episodes(tmp_path/"official.json",[
        _episode(1,0,"House beside Refined-X"),
        _episode(3,0,"Van"),
    ])
    result=audit(current,official)["val_seen"]
    assert not result["exact_byte_identical"]
    assert result["shared_ids"]==1
    assert result["changed_instructions_or_annotations"]==1
    assert result["only_current"]==1
    assert result["only_official"]==1
    assert not result["semantic_identical"]


def test_annotations_can_differ_with_identical_natural_language(tmp_path):
    local=_episodes(tmp_path/"local.json",[_episode(1,0,"The van near One Stop")])
    official=_episodes(tmp_path/"official.json",[_episode(1,0,"The van near One Stop")])
    p1=tmp_path/"processed_local.json"
    p2=tmp_path/"processed_official.json"
    p1.write_text(json.dumps({"cambridge_block_7":{"1":[{"landmarks":[]}]}}))
    p2.write_text(json.dumps({"cambridge_block_7":{"1":[{"landmarks":["One Stop"]}]}}))
    result=audit(local,official,current_processed=p1,official_processed=p2)
    assert result["val_seen_semantically_equivalent"]
    assert not result["all_supplied_semantically_equivalent"]
    assert result["processed_descriptions"]["changed_instructions_or_annotations"]==1


def test_missing_official_and_duplicate_episodes_hard_fail(tmp_path):
    current=_episodes(tmp_path/"local.json",[_episode(1,0,"House")])
    with pytest.raises(FileNotFoundError):
        audit(current,tmp_path/"not_downloaded.json")
    dupe=_episodes(tmp_path/"dup.json",[_episode(1,0,"a"),_episode(1,0,"b")])
    with pytest.raises(ValueError,match="duplicate"):
        episodes(dupe)
