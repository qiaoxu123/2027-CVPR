"""Audit Val-Seen NO-PUBLIC-LANDMARK correlation with CLOSED-LOOP failure/NE.

No rollout, no inference, no GT-selected anchors, no Test/Val-driven tuning.
Join archived FULL 2470 baseline paired CSV with the actual PUBLIC record
sent to the same policy in each episode. Recompute official public matcher
to distinguish missing name, invalid polygon and >16 overflow.

The result quantifies association, NOT causal failures "because of" anchors.
Evaluate a randomized matched causal intervention before claiming causation.

Default paths from PR25:
  --paired reports/unified_pgvb/angle_only/paired2470.csv
  --episodes reports/unified_pgvb/angle_only/episodes2470.json
  --public-records-root /.../runs/unified_angle_only_s42_20261106/val_seen_predicted_angles/belief_steps
  --public-objects /.../sbfnav-data/cityrefer_original/objects.json
  --output /.../runs/unified_angle_only_s42_20261106/val_seen_binding_audit
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter,defaultdict
from pathlib import Path
from statistics import mean

from graphnav.unified_text_ablation import public_text_record
from scripts.analyze_instruction_landmark_association import PublicNames
from scripts.prepare_unified_v2_landmark_relation_ablation import public_references


BINS=((0,20),(20,40),(40,60),(60,80),(80,100),
      (100,150),(150,200),(200,300),(300,float("inf")))


def percentiles(values):
    if not values:
        return None
    a=sorted(values)
    def q(p):
        z=(len(a)-1)*p
        lo=int(z)
        return a[lo]+(a[min(lo+1,len(a)-1)]-a[lo])*(z-lo)
    return dict(n=len(a),minimum_m=a[0],p25_m=q(.25),
                median_m=q(.5),p75_m=q(.75),p90_m=q(.9),
                p95_m=q(.95),p99_m=q(.99),maximum_m=a[-1],
                mean_m=mean(a))


def categorize_binding(record,catalog,matcher,max_landmarks):
    """REPRODUCE the FULL TEXT runtime binding with no GT inputs."""
    actual,audit=public_text_record(
        record["map_name"],record["instruction"],record["world_bounds"],
        catalog,matcher,mode="full_text",max_landmarks=max_landmarks)
    # Protect against silently auditing a different polygon pipeline:
    received=record["landmarks"]
    if [x["id"] for x in received] != [x["id"] for x in actual["landmarks"]]:
        raise ValueError("Online archived Polygon ID order and current public catalog mismatch")
    # exact polygon lists, not nearest-GT or approximate centroid matching.
    if [x["contour"] for x in received] != [x["contour"] for x in actual["landmarks"]]:
        raise ValueError("Online archived public Polygon contours mismatch")
    warnings=list(audit["warnings"])
    if received:
        status="bound_public_polygon"
    elif "all_public_polygons_exceed_capacity" in warnings:
        status="capacity_overflow_zero_anchor"
    else:
        mentions=matcher.match(record["map_name"],record["instruction"])
        if not mentions:
            status="no_public_landmark_name_match"
        else:
            status="matched_name_but_no_valid_polygon"
    mentions=matcher.match(record['map_name'],record['instruction'])
    uncapped,uncapped_warnings=public_references(
        record['map_name'],record['world_bounds'],mentions,catalog,10**9)
    audit.update(matched_mentions=len(mentions),
        matched_public_ids=list(dict.fromkeys(str(i) for m in mentions for i in m['public_ids'])),
        valid_polygon_count_before_capacity=len(uncapped),
        invalid_polygon_events=sum(w.startswith('invalid_public_polygon') for w in uncapped_warnings),
        same_name_multi_polygon=any(len(set(map(str,m['public_ids'])))>1 for m in mentions))
    return status,audit


def analyze(paired_csv,episode_manifest,public_records_root,objects_json, *,
            expected=2470,max_landmarks=16,baseline_predictions=None,annotations=None):
    data=json.loads(Path(objects_json).read_text(encoding="utf-8"))
    matcher=PublicNames(data)
    episodes=json.loads(Path(episode_manifest).read_text(encoding="utf-8"))
    with Path(paired_csv).open(newline="",encoding="utf-8") as f:
        csv_rows=list(csv.DictReader(f))
    if len(episodes)!=expected or len(csv_rows)!=expected:
        raise ValueError(f"Expected {expected}/{expected}, got {len(episodes)}/{len(csv_rows)}")
    for label,rows in [('manifest',episodes),('CSV',csv_rows)]:
        ids=[tuple(json.loads(r['episode_id']) if isinstance(r['episode_id'],str) else r['episode_id']) for r in rows]
        if len(set(ids))!=len(ids):raise ValueError(f'Duplicate {label} Episode ID')
        indices=[int(r['index']) for r in rows]
        if len(set(indices))!=len(indices):raise ValueError(f'Duplicate {label} index')
    if {int(r['index']) for r in episodes}!=set(range(expected)):
        raise ValueError('Expected complete contiguous episode indices')
    predictions=None;annotation_by_id=None
    if baseline_predictions is not None:
        pred=[json.loads(s) for s in Path(baseline_predictions).read_text().splitlines() if s.strip()]
        predictions={tuple(r['episode_id']):r for r in pred}
        if len(pred)!=expected or len(predictions)!=expected:
            raise ValueError('Duplicate or incomplete baseline predictions')
        if set(predictions)!={tuple(r['episode_id']) for r in episodes}:
            raise ValueError('Baseline Episode ID coverage mismatch')
    if annotations is not None:
        ann=json.loads(Path(annotations).read_text())
        annotation_by_id={(f"{r['area']}_block_{r['block']}",int(r['object_ids'][0]),int(r['ann_ids'][0])):r for r in ann}
        if len(ann)!=expected or len(annotation_by_id)!=expected:
            raise ValueError('Duplicate or incomplete annotations')
        if set(annotation_by_id)!={tuple(r['episode_id']) for r in episodes}:
            raise ValueError('Annotation Episode ID coverage mismatch')
    if {r["index"] for r in episodes}!={int(r["index"]) for r in csv_rows}:
        raise ValueError("Episode manifest and paired CSV indices do not align")
    manifest={int(r["index"]):r for r in episodes}
    details=[]
    seen=set()
    for row in csv_rows:
        i=int(row["index"])
        if i in seen:raise ValueError("Duplicate paired index")
        seen.add(i)
        epi=manifest[i]
        if json.loads(row["episode_id"])!=epi["episode_id"]:
            raise ValueError(f"Episode ID mismatch at index {i}")
        record_path=Path(public_records_root)/f"{i:05d}"/"public_record.json"
        if not record_path.is_file():
            raise FileNotFoundError(
                f"{record_path}: require local original online records; "
                "the GitHub report alone has no per-task landmark status"
            )
        record=json.loads(record_path.read_text(encoding="utf-8"))
        if record["map_name"]!=epi["episode_id"][0]:
            raise ValueError(f"PUBLIC record map mismatch {i}")
        if annotation_by_id is not None:
            annotation=annotation_by_id[tuple(epi['episode_id'])]
            if record['instruction']!=annotation['descriptions'][0]:
                raise ValueError(f'PUBLIC record instruction mismatch at Episode ID {epi["episode_id"]}')
        status,audit=categorize_binding(record,data,matcher,max_landmarks)
        ne=float(row["baseline_NE"])
        if not math.isfinite(ne) or ne<0:raise ValueError(f'Invalid baseline NE at {i}')
        sr=float(row['baseline_SR'])
        if sr not in (0.,1.):raise ValueError(f'Invalid baseline SR at {i}')
        success=int(sr)
        if predictions is not None:
            pred=predictions[tuple(epi['episode_id'])]
            if pred['index']!=i or pred['poses'][0]!=epi['start_pose']:
                raise ValueError(f'Baseline index/start pose mismatch at {i}')
            if pred.get('error') is not None:
                raise ValueError(f'Baseline inference error at {i}')
            if not math.isclose(pred['metrics']['NE'],ne,abs_tol=1e-8,rel_tol=0) or pred['metrics']['SR']!=sr:
                raise ValueError(f'Baseline archived SR/NE mismatch at {i}')
            if annotation_by_id is not None:
                target=annotation['target_positions'][-1]
                measured=math.dist(pred['poses'][-1][:2],target[:2])
                if not math.isclose(measured,ne,abs_tol=1e-6,rel_tol=0):
                    raise ValueError(f'Baseline final position NE mismatch at {i}')
        if (success==1)!=(ne<=20.):
            raise ValueError(f"SR and NE are inconsistent for episode {i}")
        details.append(dict(
            index=i,episode_id=epi["episode_id"],
            map_name=record["map_name"],instruction=record["instruction"],
            bound_reference_count=len(record["landmarks"]),
            reference_ids=[x["id"] for x in record["landmarks"]],
            public_match_status=status,public_warnings=audit["warnings"],
            matched_mentions=audit['matched_mentions'],matched_public_ids=audit['matched_public_ids'],
            valid_polygon_count_before_capacity=audit['valid_polygon_count_before_capacity'],
            invalid_polygon_events=audit['invalid_polygon_events'],
            same_name_multi_polygon=audit['same_name_multi_polygon'],
            baseline_SR=success,baseline_NE_m=ne,
            predicted_angle_SR=int(float(row["predicted_angle_SR"])),
            predicted_angle_NE_m=float(row["predicted_angle_NE"])))
    if len(seen)!=expected:
        raise ValueError("Incomplete Val-Seen task coverage")
    actual_paths=set(Path(public_records_root).glob('*/public_record.json'))
    expected_paths={Path(public_records_root)/f'{i:05d}'/'public_record.json' for i in seen}
    if actual_paths!=expected_paths:
        raise ValueError('Unexpected extra public records outside episode manifest')
    groups=defaultdict(list)
    for r in details:
        groups["all"].append(r)
        groups["no_anchor" if not r["bound_reference_count"] else "with_anchor"].append(r)
        groups[r["public_match_status"]].append(r)
        if not r["baseline_SR"]:
            groups["failures"].append(r)
            groups["failures_no_anchor" if not r["bound_reference_count"]
                   else "failures_with_anchor"].append(r)
    def stats(rows):
        values=[x["baseline_NE_m"] for x in rows]
        failures=[x for x in rows if not x["baseline_SR"]]
        return dict(N=len(rows),failures=len(failures),
                    failure_rate=len(failures)/len(rows) if rows else None,
                    successes=len(rows)-len(failures),
                    SR=(len(rows)-len(failures))/len(rows) if rows else None,
                    NE_all=percentiles(values),
                    NE_failures=percentiles([r["baseline_NE_m"] for r in failures]),
                    distance_bins=[dict(range_m=f"{lo}-{hi if hi!=float('inf') else '+'}",
                        n=sum(lo<=x["baseline_NE_m"]<hi for x in rows))
                        for lo,hi in BINS])
    overall={key:stats(rows) for key,rows in sorted(groups.items())}
    no_anchor_fail=len(groups["failures_no_anchor"])
    all_fail=len(groups["failures"])
    summary=dict(
        protocol="Val-Seen original full-text no-anchor correlation audit v1",
        count=len(details),outcome="baseline_full_text",
        alignment=dict(unique_episode_ids=len(details),baseline_predictions_checked=predictions is not None,
            annotation_instruction_checked=annotation_by_id is not None),
        binding_events=dict(invalid_polygon_tasks=sum(r['invalid_polygon_events']>0 for r in details),
            invalid_polygon_events=sum(r['invalid_polygon_events'] for r in details),
            capacity_overflow_tasks=sum('all_public_polygons_exceed_capacity' in r['public_warnings'] for r in details),
            same_name_multi_polygon_tasks=sum(r['same_name_multi_polygon'] for r in details)),
        no_anchor_failed=no_anchor_fail,
        all_failures=all_fail,
        no_anchor_fraction_among_failures=no_anchor_fail/all_fail if all_fail else None,
        sections=overall,
        largest_errors=sorted(
            [{k:r[k] for k in ("index","episode_id","public_match_status",
                    "bound_reference_count","baseline_SR","baseline_NE_m")}
             for r in details],key=lambda x:-x["baseline_NE_m"])[:25],
        inference_gt_access=False,
        caveat="Descriptive co-occurrence of missing public Polygon and failure. "
               "No causal conclusion without matched intervention. "
               "1,221 no-anchor cases in TRAIN are not the Val-Seen count."
    )
    return summary,details


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("paired","episodes","public-records-root","public-objects","output"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--expected",type=int,default=2470)
    p.add_argument("--max-landmarks",type=int,default=16)
    p.add_argument('--baseline-predictions',type=Path)
    p.add_argument('--annotations',type=Path)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError("Do not overwrite an existing binding audit")
    summary,details=analyze(
        a.paired,a.episodes,a.public_records_root,a.public_objects,
        expected=a.expected,max_landmarks=a.max_landmarks,
        baseline_predictions=a.baseline_predictions,annotations=a.annotations)
    a.output.mkdir(parents=True)
    (a.output/"summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    with (a.output/"per_episode.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(details[0]))
        w.writeheader()
        for x in details:
            w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in x.items()})
    print(json.dumps(dict(
        count=summary["count"],failures=summary["all_failures"],
        no_anchor_failed=summary["no_anchor_failed"],
        no_anchor_fraction_among_failures=summary["no_anchor_fraction_among_failures"],
        groups={k:dict(N=v["N"],failures=v["failures"],rate=v["failure_rate"])
                for k,v in summary["sections"].items()}),
        ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
