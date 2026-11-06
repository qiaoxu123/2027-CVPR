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
    return status,audit


def analyze(paired_csv,episode_manifest,public_records_root,objects_json, *,
            expected=2470,max_landmarks=16):
    data=json.loads(Path(objects_json).read_text(encoding="utf-8"))
    matcher=PublicNames(data)
    episodes=json.loads(Path(episode_manifest).read_text(encoding="utf-8"))
    with Path(paired_csv).open(newline="",encoding="utf-8") as f:
        csv_rows=list(csv.DictReader(f))
    if len(episodes)!=expected or len(csv_rows)!=expected:
        raise ValueError(f"Expected {expected}/{expected}, got {len(episodes)}/{len(csv_rows)}")
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
        status,audit=categorize_binding(record,data,matcher,max_landmarks)
        ne=float(row["baseline_NE"])
        success=int(float(row["baseline_SR"]))
        if (success==1)!=(ne<=20.):
            raise ValueError(f"SR and NE are inconsistent for episode {i}")
        details.append(dict(
            index=i,episode_id=epi["episode_id"],
            map_name=record["map_name"],instruction=record["instruction"],
            bound_reference_count=len(record["landmarks"]),
            reference_ids=[x["id"] for x in record["landmarks"]],
            public_match_status=status,public_warnings=audit["warnings"],
            baseline_SR=success,baseline_NE_m=ne,
            predicted_angle_SR=int(float(row["predicted_angle_SR"])),
            predicted_angle_NE_m=float(row["predicted_angle_NE"])))
    if len(seen)!=expected:
        raise ValueError("Incomplete Val-Seen task coverage")
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
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError("Do not overwrite an existing binding audit")
    summary,details=analyze(
        a.paired,a.episodes,a.public_records_root,a.public_objects,
        expected=a.expected,max_landmarks=a.max_landmarks)
    a.output.mkdir(parents=True)
    (a.output/"summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    with (a.output/"per_episode.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["index","episode_id","map_name","instruction",
                    "public_match_status","bound_reference_count",
                    "reference_ids","baseline_SR","baseline_NE_m",
                    "predicted_angle_SR","predicted_angle_NE_m","public_warnings"])
        for x in details:
            w.writerow([x["index"],json.dumps(x["episode_id"]),x["map_name"],
                x["instruction"],x["public_match_status"],x["bound_reference_count"],
                json.dumps(x["reference_ids"]),x["baseline_SR"],x["baseline_NE_m"],
                x["predicted_angle_SR"],x["predicted_angle_NE_m"],
                json.dumps(x["public_warnings"])])
    print(json.dumps(dict(
        count=summary["count"],failures=summary["all_failures"],
        no_anchor_failed=summary["no_anchor_failed"],
        no_anchor_fraction_among_failures=summary["no_anchor_fraction_among_failures"],
        groups={k:dict(N=v["N"],failures=v["failures"],rate=v["failure_rate"])
                for k,v in summary["sections"].items()}),
        ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
