"""Synthetic paired rollout comparison: exact GT-free input mutation & SR attribution."""
import json

import pytest

from scripts.analyze_public_anchor_repair_rollouts import summarize,quantile,read_run


def _output(tmp_path,mode=None):
    root=tmp_path/(mode or "baseline")
    root.mkdir()
    episodes=[dict(index=i,episode_id=["map",i,1],start_pose=[1,2,3,4])
              for i in range(4)]
    (root/"episodes.json").write_text(json.dumps(episodes))
    old_ne=[8,200,150,50]
    new_ne=old_ne[:]
    if mode=="geometry":new_ne[1]=9
    if mode=="capacity":new_ne[2]=140
    if mode=="both":new_ne[1]=9;new_ne[2]=140
    with (root/"predictions.jsonl").open("w") as f:
        for i,v in enumerate(new_ne):
            f.write(json.dumps(dict(index=i,episode_id=episodes[i]["episode_id"],
                metrics=dict(SR=float(v<=20),NE=v,OSR=0.,SPL=0.,nDTW=.1),
                error=None))+"\n")
    for i in range(4):
        path=root/"belief_steps"/f"{i:05d}"
        path.mkdir(parents=True)
        before=i in (0,3)
        applied=mode is not None and (
            i==1 and mode in ("geometry","both") or
            i==2 and mode in ("capacity","both"))
        refs=([dict(id=str(i),name="Building",object_type="Building",
                    contour=[[0,0],[1,0],[1,1],[0,1]])]
              if before or applied else [])
        rec=dict(map_name="map",instruction=f"instruction {i}",
                 world_bounds=[0,0,100,100],landmarks=refs)
        (path/"public_record.json").write_text(json.dumps(rec))
        if mode:
            audit=dict(repair_mode=mode,applied=applied,
                       original_bound=int(before),
                       repaired_bound=len(refs),GT_used=False)
            (path/"binding_repair_audit.json").write_text(json.dumps(audit))
    return root


def test_all_three_arms_compare_with_baseline_without_full_rollout(tmp_path):
    baseline=_output(tmp_path)
    geometry=_output(tmp_path,"geometry")
    capacity=_output(tmp_path,"capacity")
    both=_output(tmp_path,"both")
    data={k:v for k,v,_ in summarize(baseline,
        dict(geometry=geometry,capacity=capacity,both=both),expected=4)}
    assert data["geometry"]["fixes"]==1
    assert data["geometry"]["damages"]==0
    assert data["geometry"]["status"]["repaired"]==1
    assert data["geometry"]["far100_before"]==2
    assert data["geometry"]["far100_after"]==1
    assert data["capacity"]["status"]["repaired"]==1
    assert data["both"]["status"]["repaired"]==2
    assert data["both"]["fixes"]==1
    assert data["both"]["affected"]["N"]==2
    assert data["both"]["unaffected_trajectory_changed"]==0


def test_one_originally_bound_polygon_mutation_is_rejected(tmp_path):
    old=_output(tmp_path)
    updated=_output(tmp_path,"both")
    path=updated/"belief_steps"/"00000"/"public_record.json"
    data=json.loads(path.read_text())
    data["landmarks"][0]["id"]="other"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="originally bound task modified"):
        list(summarize(old,{"both":updated},expected=4))


def test_nonfinite_and_duplicate_episode_rejected(tmp_path):
    old=_output(tmp_path)
    path=old/"predictions.jsonl"
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["episode_id"]=["map",0,1]
    path.write_text("\n".join(json.dumps(r) for r in rows)+"\n")
    with pytest.raises(ValueError,match="Episode ID mismatch"):
        read_run(old,4)


def test_quantiles_for_both_failure_and_success_are_order_independent():
    assert quantile([100.,10.,20.,30.])["median"]==25.
    assert quantile([]) is None
