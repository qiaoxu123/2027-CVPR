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
      "sample_block_1":{
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
        records,audit=public_text_record("sample_block_1",text,[0,0,100,100],
                            public,matcher,mode="full_text",max_landmarks=16)
        path=root/f"{i:05d}"
        path.mkdir(parents=True)
        (path/"public_record.json").write_text(json.dumps(records))
        episode=["sample_block_1",i,0]
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
    old[0]["episode_id"]=["sample_block_1",99,0]
    e.write_text(json.dumps(old))
    with pytest.raises(ValueError,match="Episode ID mismatch"):
        analyze(p,e,root,objects,expected=4)
    old[0]["episode_id"]=["sample_block_1",0,0]
    e.write_text(json.dumps(old))
    (root/"00000"/"public_record.json").unlink()
    with pytest.raises(FileNotFoundError,match="require local original online"):
        analyze(p,e,root,objects,expected=4)


def test_group_quantiles_exact_small_n():
    x=percentiles([10.,20.,30.,40.])
    assert x["median_m"]==25.
    assert x["p75_m"]==32.5
    assert x["maximum_m"]==40.


def test_rejects_duplicate_episode_ids_even_with_unique_indices(tmp_path):
    p,e,root,objects=_data(tmp_path)
    eps=json.loads(e.read_text());eps[1]['episode_id']=eps[0]['episode_id']
    e.write_text(json.dumps(eps))
    with pytest.raises(ValueError,match='Duplicate.*Episode ID'):
        analyze(p,e,root,objects,expected=4)


@pytest.mark.parametrize('value', ['nan', 'inf', '-1'])
def test_rejects_invalid_ne(tmp_path,value):
    p,e,root,objects=_data(tmp_path)
    rows=list(csv.DictReader(p.open()));rows[1]['baseline_NE']=value
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    with pytest.raises(ValueError,match='Invalid.*NE'):
        analyze(p,e,root,objects,expected=4)


def test_rejects_fractional_sr(tmp_path):
    p,e,root,objects=_data(tmp_path)
    rows=list(csv.DictReader(p.open()));rows[1]['baseline_SR']='0.5'
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    with pytest.raises(ValueError,match='Invalid.*SR'):
        analyze(p,e,root,objects,expected=4)


def _provenance(tmp_path):
    p,e,root,obj=_data(tmp_path)
    eps=json.loads(e.read_text());rows=list(csv.DictReader(p.open()))
    ann=[];pred=[]
    for ep,row in zip(eps,rows):
        i=ep['index'];rec=json.loads((root/f'{i:05d}'/'public_record.json').read_text())
        ann.append(dict(area='sample',block=1,object_ids=[i],ann_ids=[0],
                        descriptions=[rec['instruction']],target_positions=[[float(row['baseline_NE']),0,0]]))
        pred.append(dict(index=i,episode_id=ep['episode_id'],poses=[ep['start_pose'],[0,0,50,0]],
                         metrics=dict(NE=float(row['baseline_NE']),SR=float(row['baseline_SR'])),error=None))
    a=tmp_path/'ann.json';a.write_text(json.dumps(ann))
    b=tmp_path/'pred.jsonl';b.write_text('\n'.join(json.dumps(r) for r in pred))
    return p,e,root,obj,a,b


def test_strict_provenance_accepts_reordered_predictions(tmp_path):
    p,e,root,obj,a,b=_provenance(tmp_path)
    b.write_text('\n'.join(reversed(b.read_text().splitlines())))
    s,rows=analyze(p,e,root,obj,expected=4,baseline_predictions=b,annotations=a)
    assert s['alignment']['baseline_predictions_checked']
    assert s['binding_events']['invalid_polygon_tasks']==1


def test_rejects_same_map_record_swap(tmp_path):
    p,e,root,obj,a,b=_provenance(tmp_path)
    x=root/'00000/public_record.json';y=root/'00003/public_record.json'
    left=x.read_text();x.write_text(y.read_text());y.write_text(left)
    with pytest.raises(ValueError,match='instruction mismatch'):
        analyze(p,e,root,obj,expected=4,baseline_predictions=b,annotations=a)


@pytest.mark.parametrize('field', ['NE','start','end','id'])
def test_rejects_corrupt_baseline_provenance(tmp_path,field):
    p,e,root,obj,a,b=_provenance(tmp_path)
    pred=[json.loads(s) for s in b.read_text().splitlines()]
    if field=='NE':pred[0]['metrics']['NE']+=1
    if field=='start':pred[0]['poses'][0][0]+=1
    if field=='end':pred[0]['poses'][-1][0]+=1
    if field=='id':pred[0]['episode_id']=pred[1]['episode_id']
    b.write_text('\n'.join(map(json.dumps,pred)))
    with pytest.raises(ValueError):
        analyze(p,e,root,obj,expected=4,baseline_predictions=b,annotations=a)


@pytest.mark.parametrize('part',['id','contour','extra'])
def test_rejects_binding_geometry_changes_and_extra_records(tmp_path,part):
    p,e,root,obj=_data(tmp_path)
    path=root/'00000/public_record.json';record=json.loads(path.read_text())
    if part=='id':record['landmarks'][0]['id']='999'
    if part=='contour':record['landmarks'][0]['contour'][0][0]+=.01
    if part=='extra':
        (root/'99999').mkdir();(root/'99999/public_record.json').write_text(json.dumps(record))
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):analyze(p,e,root,obj,expected=4)


def test_capacity_overflow_retains_task_and_tracks_all_candidates(tmp_path):
    from scripts.analyze_val_seen_binding_failures import categorize_binding
    catalog={'m':{str(i):dict(name='One Stop',object_type='Building',
        contour=[[i,0],[i+.5,0],[i+.5,1],[i,1]]) for i in range(17)}}
    matcher=PublicNames(catalog)
    record,_=public_text_record('m','near One Stop',[0,0,100,100],catalog,matcher,mode='full_text',max_landmarks=16)
    status,audit=categorize_binding(record,catalog,matcher,16)
    assert record['landmarks']==[]
    assert status=='capacity_overflow_zero_anchor'
    assert audit['valid_polygon_count_before_capacity']==17
    assert audit['same_name_multi_polygon']
