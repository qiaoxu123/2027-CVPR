"""Paired real Val-Seen PUBLIC binding intervention audit; NO TRAIN/NO rollout.

Input: full baseline run and 3 completed frozen checkpoint intervention runs.
All 2470 episodes and manifest IDs must align; missing/incomplete runs FAIL.
Read the actual model-bound Polygon IDs and complete contours per episode.
Check that originally bound tasks remain exactly unchanged as intended.
Report fixes/breaks, SR/NE, >=100m far failures and risk by affected subset.

Causality: same-policy input change is an intervention, not a clean claim about
all "no name" CityNav navigation errors or unseen generalization. Since
Val-Seen errors were already examined, this experiment is diagnostic.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def read_run(folder,expected):
    folder=Path(folder)
    manifest=json.loads((folder/"episodes.json").read_text(encoding="utf-8"))
    if len(manifest)!=expected:
        raise ValueError(f"{folder}: expected {expected} manifest entries")
    id_by_idx={int(x["index"]):x for x in manifest}
    if len(id_by_idx)!=expected or set(id_by_idx)!=set(range(expected)):
        raise ValueError("Missing/duplicated/noncontinuous manifest index")
    rows=[json.loads(line) for line in (folder/"predictions.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    pred={int(x["index"]):x for x in rows}
    if len(pred)!=expected or len(rows)!=expected or set(pred)!=set(id_by_idx):
        raise ValueError(f"{folder}: incomplete/duplicate predictions")
    for i in range(expected):
        x=pred[i];m=id_by_idx[i]
        if list(x["episode_id"])!=list(m["episode_id"]):
            raise ValueError(f"{folder}: Episode ID mismatch {i}")
        for k in ("SR","NE","OSR","SPL","nDTW"):
            if k not in x["metrics"] or not math.isfinite(float(x["metrics"][k])):
                raise ValueError(f"{folder}: invalid metric {k}, episode {i}")
        if not 0<=float(x["metrics"]["SR"])<=1:
            raise ValueError("Invalid SR")
        if not x.get("error") in (None,""):
            raise ValueError(f"{folder}: runtime inference error in {i}")
    return id_by_idx,pred


def quantile(values):
    a=np.asarray(values,dtype=float)
    if not len(a):return None
    return dict(n=len(a),mean=float(a.mean()),median=float(np.quantile(a,.5)),
                p90=float(np.quantile(a,.9)),p95=float(np.quantile(a,.95)),
                p99=float(np.quantile(a,.99)),max=float(a.max()))


def _record(folder,i):
    return json.loads((Path(folder)/"belief_steps"/f"{i:05d}"/"public_record.json").read_text())


def _audit(folder,i):
    return json.loads((Path(folder)/"belief_steps"/f"{i:05d}"/
                       "binding_repair_audit.json").read_text())


def _signature(record):
    return [(str(x["id"]),str(x["name"]),x["object_type"],x["contour"])
            for x in record["landmarks"]]


def _metrics(rows):
    n=len(rows)
    if not n:return dict(N=0)
    success=sum(int(r["new_SR"]) for r in rows)
    near=np.asarray([r["new_NE_m"] for r in rows])
    return dict(N=n,success=success,failures=n-success,SR=success/n,
                NE=quantile(near),
                far100_failures=int(sum(not r["new_SR"] and r["new_NE_m"]>=100
                                        for r in rows)))


def _cluster_bootstrap(rows,seed=42,iterations=2000):
    if not rows:return None
    clusters=defaultdict(list)
    for r in rows:
        e=r["episode_id"]
        clusters[(str(e[0]),str(e[1]))].append(r)
    data=list(clusters.values());rng=np.random.default_rng(seed)
    changes=[]
    for _ in range(iterations):
        sampled=rng.integers(0,len(data),len(data))
        chosen=[r for ix in sampled for r in data[int(ix)]]
        changes.append(sum(r["new_SR"]-r["old_SR"] for r in chosen)/len(chosen))
    return [float(np.quantile(changes,.025)),float(np.quantile(changes,.975))]


def summarize(baseline,arms,expected=2470):
    base_manifest,base=read_run(baseline,expected)
    all_rows={}
    for mode,folder in arms.items():
        manifest,pred=read_run(folder,expected)
        if manifest!=base_manifest:
            raise ValueError(f"{mode}: episode/start-pose manifest changed")
        sample=[];seen_modes=Counter()
        for i in range(expected):
            old=_record(baseline,i)
            new=_record(folder,i)
            before=_signature(old)
            after=_signature(new)
            if old["instruction"]!=new["instruction"] or old["map_name"]!=new["map_name"]:
                raise ValueError(f"{mode}: instruction/map modified {i}")
            if old["world_bounds"]!=new["world_bounds"]:
                raise ValueError(f"{mode}: world bounds modified {i}")
            audit=_audit(folder,i)
            if audit["repair_mode"]!=mode:
                raise ValueError(f"{mode}: unexpected intervention mode {i}")
            if audit.get("GT_used") or audit.get("GT_used_to_bind") or audit.get("distance_used_to_select"):
                raise ValueError(f"{mode}: source says GT used for binding")
            if audit.get("applied") != bool(not before and after):
                raise ValueError(f"{mode}: applied flag does not match input {i}")
            if before and before!=after:
                raise ValueError(f"{mode}: originally bound task modified: {i}")
            if not before and len(after)>16:
                raise ValueError(f"{mode}: model capacity exceeded at {i}")
            # Plain old inputs must remain the same; the original baseline
            # is a frozen CONTROL, so any changed outcomes on these episodes
            # require a repeatability investigation.
            status="repaired" if not before and after else \
                   "unresolved" if not before else "originally_bound"
            seen_modes[status]+=1
            old_sr=int(base[i]["metrics"]["SR"])
            new_sr=int(pred[i]["metrics"]["SR"])
            old_ne=float(base[i]["metrics"]["NE"])
            new_ne=float(pred[i]["metrics"]["NE"])
            sample.append(dict(index=i,episode_id=base_manifest[i]["episode_id"],
                status=status,old_SR=old_sr,new_SR=new_sr,old_NE_m=old_ne,
                new_NE_m=new_ne,old_polygon_count=len(before),
                new_polygon_count=len(after),old_ids=[x[0] for x in before],
                new_ids=[x[0] for x in after],details=audit))
        changed=[r for r in sample if r["status"]=="repaired"]
        no_anchor=[r for r in sample if r["status"]!="originally_bound"]
        distant=[r for r in sample if not r["old_SR"] and r["old_NE_m"]>=100]
        diff=[r for r in sample if r["old_SR"]!=r["new_SR"]]
        fixes=sum(r["new_SR"]>r["old_SR"] for r in sample)
        damages=sum(r["new_SR"]<r["old_SR"] for r in sample)
        sr_new=sum(r["new_SR"] for r in sample)/expected
        sr_old=sum(r["old_SR"] for r in sample)/expected
        result=dict(episodes=expected,mode=mode,status=dict(seen_modes),
            baseline_SR=sr_old,intervention_SR=sr_new,
            delta_SR_pp=100*(sr_new-sr_old),fixes=fixes,damages=damages,
            net=fixes-damages,full_NEm=quantile([r["new_NE_m"] for r in sample]),
            baseline_NEm=quantile([r["old_NE_m"] for r in sample]),
            affected=_metrics(changed),originally_no_anchor=_metrics(no_anchor),
            originally_far100=_metrics(distant),
            far100_before=sum(not r["old_SR"] and r["old_NE_m"]>=100 for r in sample),
            far100_after=sum(not r["new_SR"] and r["new_NE_m"]>=100 for r in sample),
            paired_cluster_SR_delta95CI_fraction=_cluster_bootstrap(sample),
            affected_indices=[r["index"] for r in changed],
            changed_SR_indices=[r["index"] for r in diff],
            unaffected_trajectory_changed=sum(
                r["status"]!="repaired" and abs(r["old_NE_m"]-r["new_NE_m"])>1.e-3
                for r in sample),
            note="Originally bound episodes have unchanged public model inputs. "
                 "If their trajectories change, audit deterministic rollout before "
                 "claiming causal effects.",
        )
        all_rows[mode]=sample
        yield mode,result,sample


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--geometry",type=Path,required=True)
    p.add_argument("--capacity",type=Path,required=True)
    p.add_argument("--both",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--expected",type=int,default=2470)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError("Refuse overwrite of paired repair analysis")
    result={}
    rows=[]
    for name,summary,items in summarize(args.baseline,
             dict(geometry=args.geometry,capacity=args.capacity,both=args.both),
             expected=args.expected):
        result[name]=summary
        for r in items:
            rows.append(dict(mode=name,**r))
    args.output.mkdir(parents=True)
    (args.output/"summary.json").write_text(json.dumps(result,indent=2,
        ensure_ascii=False)+"\n",encoding="utf-8")
    with (args.output/"per_episode.jsonl").open("w",encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r,ensure_ascii=False)+"\n")
    print(json.dumps({
        k:{field:v[field] for field in
           ("mode","delta_SR_pp","fixes","damages","far100_before",
            "far100_after","status","unaffected_trajectory_changed")}
        for k,v in result.items()},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
