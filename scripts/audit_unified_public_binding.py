"""Reproducible TRAIN binding audit and optional PUBLIC-BLIND full-text JSONL.

Compare the preprocessed source references (derived from processed_descriptions
looked up by target object ID + annotation ID) against the EXACT public-only
text/name/Polygon binding used by PR23/PR25/PR26 Val-Seen.

No target GT, target IDs, human poses or future frames are consulted to select
any reference. Target_xy is copied as an unchanged TRAIN LOSS LABEL ONLY.
Read the full 21,878 Revised TRAIN JSONL; compare per-query in original order.

Examples:
python -m scripts.audit_unified_public_binding \
 --train-jsonl /path/train_revised.jsonl \
 --public-objects /path/objects.json \
 --output /path/binding_audit \
 --emit-blind-train /path/train_public_blind.jsonl

The emitted file is NOT written unless the flag is passed. Train all arms
on the same emitted file to remove original annotation-linked binder shift.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


from graphnav.unified_direct_bearing_loss import predicate_ids_for_record
from scripts.analyze_instruction_landmark_association import PublicNames
from scripts.prepare_unified_v2_landmark_relation_ablation import public_references


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ids(refs):
    return tuple(str(ref["id"]) for ref in refs)


def _category(before, after):
    a,b=set(before),set(after)
    if not a and not b:return "both_empty"
    if before==after:return "identical_ids_and_order"
    if a==b:return "same_ids_different_order"
    if not a and b:return "blind_only"
    if a and not b:return "preprocessed_only"
    if a<=b:return "preprocessed_subset"
    if b<=a:return "blind_subset"
    if not a.isdisjoint(b):return "partial_overlap"
    return "disjoint_nonempty"


def inspect_rows(rows,catalog,max_landmarks):
    matcher=PublicNames(catalog)
    summary=Counter()
    sample=defaultdict(list)
    details=[]
    blind_records=[]
    for i,row in enumerate(rows):
        if row["map_name"] not in catalog:
            raise KeyError("Missing PUBLIC catalog map "+row["map_name"])
        public_map=catalog[row["map_name"]]
        mentions=matcher.match(row["map_name"],row["instruction"])
        refs,warnings=public_references(
            row["map_name"],row["world_bounds"],mentions,catalog,max_landmarks)
        pre=row.get("landmarks",[])
        before,after=_ids(pre),_ids(refs)
        category=_category(before,after)
        pre_elig=predicate_ids_for_record(row,public_map,max_landmarks)[1]>=0
        blind=copy.deepcopy(row)
        blind["landmarks"]=refs
        blind_elig=predicate_ids_for_record(blind,public_map,max_landmarks)[1]>=0
        pre_set=set(before);blind_set=set(after)
        all_mentions={str(x) for m in mentions for x in m["public_ids"]}
        signature=dict(
            index=i,map_name=row["map_name"],status=category,
            source_preprocessed_count=len(pre),blind_public_count=len(refs),
            source_ids=list(before),blind_ids=list(after),
            source_only_ids=sorted(pre_set-blind_set),
            public_blind_only_ids=sorted(blind_set-pre_set),
            source_ids_not_mentioned_in_full_public_catalog_match=sorted(pre_set-all_mentions),
            explicit_public_name_mentions=len(mentions),
            warnings=warnings,
            direct_cardinal_source=int(pre_elig),
            direct_cardinal_blind=int(blind_elig),
        )
        details.append(signature)
        blind_records.append(blind)
        summary["queries"]+=1
        summary["status_"+category]+=1
        summary["source_preprocessed_polygon_slots"]+=len(pre)
        summary["blind_public_polygon_slots"]+=len(refs)
        summary["source_no_public_anchor"]+=not bool(pre)
        summary["blind_no_public_anchor"]+=not bool(refs)
        summary["had_explicit_public_name_mentions"]+=bool(mentions)
        summary["source_reference_ids_not_text_matched"]+=bool(pre_set-all_mentions)
        summary["source_only_polygon_ids"]+=len(pre_set-blind_set)
        summary["blind_only_polygon_ids"]+=len(blind_set-pre_set)
        summary["source_eligible_direct_cardinal"]+=pre_elig
        summary["blind_eligible_direct_cardinal"]+=blind_elig
        summary["cardinal_eligibility_gained_after_blind"]+=blind_elig and not pre_elig
        summary["cardinal_eligibility_lost_after_blind"]+=pre_elig and not blind_elig
        summary["distinct_public_ids_both_have"]+=len(pre_set&blind_set)
        if warnings:summary["queries_with_public_binding_warnings"]+=1
        for reason in warnings:summary["warning_"+reason]+=1
        if len(sample[category])<8:sample[category].append(signature)
    return dict(summary),details,dict(sample),blind_records


def run(train_path,public_path,output_dir,*,max_landmarks=16,
        expected_records=21878,emit_blind_train=None):
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Binding audit output directory must be empty")
    if emit_blind_train is not None and emit_blind_train.exists():
        raise FileExistsError("Refuse to overwrite existing TRAIN")
    rows=[json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()
          if line.strip()]
    if len(rows)!=expected_records:
        raise ValueError(f"Expected {expected_records} source TRAIN rows, got {len(rows)}")
    if {r.get("dataset_version") for r in rows}!={"revised"} or \
       {r.get("match_mode") for r in rows}!={"fuzzy"}:
        raise ValueError("Source must be complete Revised/fuzzy TRAIN")
    if any("teacher_future" in r or "uav_pose" in r for r in rows):
        raise ValueError("Audit requires query-level static-only dataset")
    catalog=json.loads(public_path.read_text(encoding="utf-8"))
    summary,details,samples,blind=inspect_rows(rows,catalog,max_landmarks)
    output_dir.mkdir(parents=True,exist_ok=True)
    (output_dir/"detail.jsonl").write_text(
        "\n".join(json.dumps(r,ensure_ascii=False) for r in details)+"\n",
        encoding="utf-8")
    (output_dir/"example_differences.json").write_text(
        json.dumps(samples,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    result=dict(
        protocol="original-preprocessed-vs-instruction-public-blind-v1",
        source_sha256=sha256(train_path),public_catalog_sha256=sha256(public_path),
        match_algorithm="PR23 PublicNames.match + public_references",
        source_algorithm="processed_descriptions[target_id][ann_id] -> resolve_references fuzzy",
        source_target_GT_used_to_select_polygons="Original processed annotation linkage; YES target_id/ann_id lookup",
        blind_target_GT_used_to_select_polygons=False,
        does_not_assume_online_source_refs_equal_training_refs=True,
        max_landmarks=max_landmarks,summary=summary,
    )
    if emit_blind_train is not None:
        emit_blind_train.parent.mkdir(parents=True,exist_ok=True)
        with emit_blind_train.open("w",encoding="utf-8") as f:
            for row in blind:f.write(json.dumps(row,ensure_ascii=False)+"\n")
        prior_prov=Path(str(train_path)+".provenance.json")
        source_prov=json.loads(prior_prov.read_text()) if prior_prov.exists() else {}
        provenance=dict(
            output_sha256=sha256(emit_blind_train),
            dataset_version_declared="revised",
            match_mode="fuzzy",static_only=True,
            records_generated=len(rows),episodes_skipped=0,
            base_train_sha256=sha256(train_path),
            public_catalog_sha256=sha256(public_path),
            binding_algorithm="instruction_public_blind_PR23",
            annotation_selected_source_references_removed=True,
            audit_protocol=result["protocol"],
            source_sha256=source_prov.get("source_sha256"),
        )
        Path(str(emit_blind_train)+".provenance.json").write_text(
            json.dumps(provenance,indent=2,ensure_ascii=False)+"\n",
            encoding="utf-8")
        result["optional_blind_train"]=dict(
            path=str(emit_blind_train),sha256=sha256(emit_blind_train),
            original_instruction_and_label_unchanged=True,
            only_landmarks_rebound=True,
        )
    (output_dir/"summary.json").write_text(
        json.dumps(result,indent=2,ensure_ascii=False)+"\n",
        encoding="utf-8")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-jsonl",required=True,type=Path)
    p.add_argument("--public-objects",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--max-landmarks",type=int,default=16)
    p.add_argument("--expected-records",type=int,default=21878)
    p.add_argument("--emit-blind-train",type=Path,default=None)
    a=p.parse_args()
    report=run(a.train_jsonl,a.public_objects,a.output,
               max_landmarks=a.max_landmarks,expected_records=a.expected_records,
               emit_blind_train=a.emit_blind_train)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
