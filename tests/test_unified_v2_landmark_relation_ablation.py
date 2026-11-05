"""Tests ensure main Unified-v2 paired inputs, not a standalone surrogate."""
import json

import pytest

from graphnav.candidate_geometry import FullCandidateDataset
from scripts.prepare_unified_v2_landmark_relation_ablation import (
    extract_pairs,normalized_instruction,prepare,
)
from scripts.analyze_instruction_landmark_association import PublicNames


def _catalog():
    poly=[[0,0],[20,0],[20,20],[0,20]]
    return {
        "cambridge_block_1":{
            "1":dict(name="One Stop",object_type="Building",contour=poly),
            "2":dict(name="One Stop",object_type="Building",
                     contour=[[25,0],[35,0],[35,20],[25,20]]),
        },
        "birmingham_block_2":{
            "3":dict(name="Blackwell Street",object_type="TrafficRoad",
                     contour=[[50,10],[80,10],[80,25],[50,25]])
        }
    }


def _row(text,mapname):
    # Deliberately insert a GT-selected processed reference that MUST NOT
    # reach any of the newly prepared model input files.
    return dict(
        map_name=mapname,instruction=text,
        dataset_version="revised",match_mode="fuzzy",
        world_bounds=[0,0,100,100],physical_map_sizes=[100,100],
        landmarks=[dict(id="target_leak",name="Wrong GT Landmark",
                        contour=[[.5,.5],[.6,.5],[.6,.6],[.5,.6]],
                        object_type="Building")],
        target_xy=[.7,.3],
    )


def test_geometry_terms_are_only_adjacent_to_a_public_name():
    catalog=_catalog();names=PublicNames(catalog)
    text="The blue car in a parking lot behind One Stop."
    pairs=extract_pairs(text,names.match("cambridge_block_1",text))
    assert [(p["name"],p["phrase"]) for p in pairs]==[("One Stop","behind")]
    assert normalized_instruction(pairs,"landmark_relation")=="behind One Stop"
    assert "car" not in normalized_instruction(pairs,"landmark_relation")
    assert "parking" not in normalized_instruction(pairs,"landmark_relation")


def test_cardinal_direction_compound_and_negation_are_conservative():
    names=PublicNames(_catalog())
    a="The blue car north-east of One Stop."
    b="The blue car not north of One Stop."
    aa=extract_pairs(a,names.match("cambridge_block_1",a))
    bb=extract_pairs(b,names.match("cambridge_block_1",b))
    assert normalized_instruction(aa,"landmark_relation")=="northeast of One Stop"
    assert normalized_instruction(bb,"landmark_relation")=="One Stop"


def test_paired_v2_datasets_replace_all_language_paths_and_gt_anchors(tmp_path):
    catalog=_catalog()
    obj=tmp_path/"objects.json";obj.write_text(json.dumps(catalog))
    rows=[
        _row("The blue car north of One Stop.","cambridge_block_1"),
        _row("The white building next to Blackwell Street.","birmingham_block_2"),
        _row("Find the blue vehicle with nothing named.","birmingham_block_2"),
    ]
    source=tmp_path/"train.jsonl"
    source.write_text("".join(json.dumps(r)+"\n" for r in rows))
    out=tmp_path/"prepared"
    report=prepare(source,obj,out,expected=3)
    assert report["all_queries"]==3
    files={mode:[json.loads(line) for line in
                 (out/(mode+".jsonl")).read_text().splitlines()]
           for mode in ("full_text","landmark_relation","landmark_only")}
    assert files["landmark_relation"][0]["instruction"]=="north of One Stop"
    assert files["landmark_only"][0]["instruction"]=="One Stop"
    assert files["landmark_relation"][2]["instruction"]=="[UNK]"
    for mode,records in files.items():
        assert len(records)==3
        assert all(r["landmarks"][0]["name"]=="One Stop"
                   for r in records[:1])
        assert all("Wrong GT Landmark" not in json.dumps(r["landmarks"])
                   for r in records)
        assert all(r["target_xy"]==[.7,.3] for r in records)
        assert len(records[0]["landmarks"])==2 # ALL homonymous public parts
        assert records[2]["landmarks"]==[]
    assert files["landmark_relation"][0]["landmarks"]==files["full_text"][0]["landmarks"]
    # Model-facing candidate geometry and graph parser both use the new
    # instruction. Old target descriptions cannot leak via graph relations.
    data=FullCandidateDataset(out/"landmark_relation.jsonl",
                              deduplicate_queries=False)
    item=data[0]
    assert item["text"]=="north of One Stop"
    assert "blue" not in json.dumps(item["node_names"]).lower()
    assert "target_leak" not in json.dumps(item["node_names"]).lower()


def test_missing_or_partial_data_and_overwrite_fail(tmp_path):
    obj=tmp_path/"public.json";obj.write_text(json.dumps(_catalog()))
    input_file=tmp_path/"source.jsonl";input_file.write_text(
        json.dumps(_row("car behind One Stop","cambridge_block_1"))+"\n")
    with pytest.raises(ValueError,match="Expected complete"):
        prepare(input_file,obj,tmp_path/"out",expected=21878)
    prepare(input_file,obj,tmp_path/"out",expected=1)
    with pytest.raises(FileExistsError):
        prepare(input_file,obj,tmp_path/"out",expected=1)
