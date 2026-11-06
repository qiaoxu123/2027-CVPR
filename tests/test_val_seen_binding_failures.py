"""Offline real Val-Seen binding/failure audit tests (small synthetic manifest)."""
import csv
import json
from pathlib import Path

import pytest

from graphnav.unified_text_ablation import public_text_record
from scripts.analyze_instruction_landmark_association import PublicNames
from scripts.analyze_val_seen_binding_failures import analyze,percentiles


def _data(tmp_path):
    public={
      "sample_map":{
        "10":dict(name="One Stop",object_type="Building",
                  contour=[[10,10],[20,10],[20,20],[10,20]]),
        "20":dict(name="Broken Stop",object_type="Building",
                  contour=[[1,1],[2,2]]),
      }
    }
    obj=tmp_path/"objects.json"
    obj.write_text(json.dumps(public))
    matcher=PublicNames(public)
    instructions=["The car north of One Stop", "The blue car by the unnamed park",
                  "The car south of Broken Stop", "The car near One Stop"]
    episodes=[]
    csvrows=[]
    root=tmp_path/"belief_steps"
    for i,text in enumerate(instructions):
        records,audit=public_text_record("sample_map",text,[0,0,100,100],
                            public,matcher,mode="full_text",max_landmarks=16)
        path=root/f"{i:05d}"
        path.mkdir(parents=True)
        (path/"public_record.json").write_text(json.dumps(records))
        episode=["sample_map",i,0]
        episodes.append(dict(index=i,episode_id=episode,start_pose=[0,0,50,0]))
        ne=[12.0,30.0,300.0,385.8][i]
        csvrows.append(dict(index=i,episode_id=json.dumps(episode),
                            baseline_SR=float(ne<=20),predicted_angle_SR=float(ne<=20),
                            baseline_NE=ne,predicted_angle_NE=ne,termination="policy_stop"))
    eps=tmp_path/"episodes.json";eps.write_text(json.dumps(episodes))
    csvpath=tmp_path/"paired.csv"
    with csvpath.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0]))
        w.writeheader();w.writerows(csvrows)
    return csvpath,eps,root,obj


def test_no_anchor_val_seen_failure_join_and_range(tmp_path):
    p,e,root,objects=_data(tmp_path)
    summary,rows=analyze(p,e,root,objects,expected=4)
    assert summary["all_failures"]==3
    assert summary["no_anchor_failed"]==2
    assert summary["sections"]["no_anchor"]["N"]==2
    assert summary["sections"]["no_anchor"]["failure_rate"]==1.
    assert summary["sections"]["all"]["NE_all"]["maximum_m"]==385.8
    assert summary["largest_errors"][0]["index"]==3
    assert rows[0]["public_match_status"]=="bound_public_polygon"
    assert rows[1]["public_match_status"]=="no_public_landmark_name_match"
    assert rows[2]["public_match_status"]=="matched_name_but_no_valid_polygon"


def test_refuses_wrong_order_and_missing_public_records(tmp_path):
    p,e,root,objects=_data(tmp_path)
    old=json.loads(e.read_text())
    old[0]["episode_id"]=["sample_map",2,0]
    e.write_text(json.dumps(old))
    with pytest.raises(ValueError,match="Episode ID mismatch"):
        analyze(p,e,root,objects,expected=4)
    old[0]["episode_id"]=["sample_map",0,0]
    e.write_text(json.dumps(old))
    (root/"00000"/"public_record.json").unlink()
    with pytest.raises(FileNotFoundError,match="require local original online"):
        analyze(p,e,root,objects,expected=4)


def test_group_quantiles_exact_small_n():
    x=percentiles([10.,20.,30.,40.])
    assert x["median_m"]==25.
    assert x["p75_m"]==32.5
    assert x["maximum_m"]==40.
