"""Prepare PAIRED UnifiedPGVB-v2 datasets by changing ONLY language preprocessing.

This is NOT the PR22 standalone model. Train all arms with the ORIGINAL
UnifiedPGVB-v2 Geometry + CandidateAnchorRelation + PGVB checkpoint loader.

All language paths agree:
- raw instruction -> instruction BERT -> Geometry and Relation;
- raw/reduced instruction -> public spatial relation graph/rules;
- public names -> node-name BERT embeddings.

Inputs per query are public-name-matched CityRefer Polygon(s) from the public
objects catalog, not target-selected "landmarks" in the processed source.

Generate paired files for three INDEPENDENT fully-trained Unified-v2 arms:
  full_text:         original instruction
  landmark_relation: only <geometric phrase> <public landmark name>
  landmark_only:     only <public landmark name>

Preserve target_xy ONLY as a training label. No GT-selected name/reference.
All 21,878 tasks remain in exact input order, including no-name tasks.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import shapely

from graphnav.geometry import normalize_xy
from scripts.analyze_instruction_landmark_association import PublicNames

# Local, whole-phrase, immediately-before-public-name extraction.
# No separate target objects, appearance adjectives or indirect hop graph.
_CARDINAL=r"(?:north(?:[-\s]+)?east|north(?:[-\s]+)?west|south(?:[-\s]+)?east|south(?:[-\s]+)?west|north|south|east|west)"
_PHRASES=[
    rf"(?:to\s+the\s+)?{_CARDINAL}\s+of",
    r"(?:on\s+)?the\s+(?:north|south|east|west|left|right)\s+(?:hand\s+)?side\s+of",
    r"(?:to\s+the\s+)?(?:left|right)\s+of",
    r"(?:behind|in\s+back\s+of|in\s+front\s+of|ahead\s+of|at\s+the\s+rear\s+of)",
    r"(?:next\s+to|adjacent\s+to|close\s+to|near|beside|along|alongside)",
    r"(?:inside|within|outside(?:\s+of)?|on|in)",
    r"(?:opposite(?:\s+to)?|across\s+from|across\s+the\s+road\s+from)",
]
_ATTACHED=re.compile(r"(?<!\w)("+ "|".join(_PHRASES) + r")\s+(?:the\s+)?$",re.I)
_NEGATED=re.compile(r"(?:\bnot\b|\bneither\b|\bexcept\b)\s*$",re.I)
_VALID_MODES=("full_text","landmark_relation","landmark_only")


def clean_phrase(phrase):
    # Keep same relative word semantics; normalize spelling only.
    return re.sub(r"\b(north|south)[-\s]+(east|west)\b",
                  lambda m:m.group(1).lower()+m.group(2).lower(),
                  phrase.lower())


def extract_pairs(text, mentions):
    """Lexically bind geometry to its visible named landmark span.

    No claim that target is in the direct relation: car in a parking lot
    behind One Stop gives 'behind One Stop', NOT a validated target->anchor
    edge. Do not fake a direction frame for behind/left/right.
    """
    pairs=[]
    for mention in mentions:
        name=mention["text"]
        preceding=text[:mention["start"]]
        found=_ATTACHED.search(preceding)
        word=""
        if found and not _NEGATED.search(preceding[:found.start()]):
            word=clean_phrase(found.group(1).strip())
        pairs.append(dict(name=name,phrase=word,
                          span=[int(mention["start"]),int(mention["end"])],
                          public_ids=list(mention["public_ids"])))
    return pairs


def public_references(map_name, bounds, mentions, catalog, max_landmarks):
    """No target-coordinate ranking; preserve every mapped public ID."""
    refs=[];seen=set();failures=[]
    for mention in mentions:
        for oid in mention["public_ids"]:
            oid=str(oid)
            if oid in seen:continue
            seen.add(oid)
            if oid not in catalog[map_name]:
                failures.append("catalog_missing_object_id")
                continue
            public=catalog[map_name][oid]
            polygon=np.asarray(public.get("contour",[]),dtype=np.float64)
            if polygon.ndim!=2 or polygon.shape[1]!=2 or len(polygon)<3 or not np.isfinite(polygon).all():
                failures.append("invalid_public_polygon_coordinates");continue
            polygon_shape=shapely.Polygon(polygon)
            if not polygon_shape.is_valid or polygon_shape.area<=0:
                failures.append("invalid_public_polygon_shape");continue
            contours=normalize_xy(polygon,bounds)
            refs.append(dict(id=oid,name=str(public["name"]),
                             object_type=public.get("object_type","Other"),
                             aliases=list(public.get("aliases",[])),
                             contour=contours.tolist()))
    if len(refs)>max_landmarks:
        failures.append("all_public_polygons_exceed_capacity")
        return [],failures
    return refs,failures


def normalized_instruction(pairs,mode):
    if mode not in _VALID_MODES:raise ValueError(mode)
    if mode=="full_text":raise ValueError("full_text must preserve original string")
    segments=[];seen=set()
    for pair in pairs:
        # Ignore duplicated exact name+predicate when the text repeats the
        # same landmark. Do not merge distinct predicates.
        s=(pair["phrase"]+" " if pair["phrase"] and mode=="landmark_relation" else "") +pair["name"]
        key=s.casefold()
        if key not in seen:
            segments.append(s);seen.add(key)
    # Separator is punctuation only, not any additional source-language word.
    # Empty is represented by the stable BERT-only [UNK] sentinel, to avoid
    # an invalid all-false token mask in original Geometry.
    return "; ".join(segments) if segments else "[UNK]"


def prepare(source_path,catalog_path,out_dir,*,expected=21878,
            max_landmarks=16):
    if out_dir.exists():
        raise FileExistsError("Refuse to overwrite ablation outputs")
    source=Path(source_path).read_text(encoding="utf-8").splitlines()
    rows=[json.loads(x) for x in source if x.strip()]
    if len(rows)!=expected:raise ValueError(f"Expected complete {expected} TRAIN queries, got {len(rows)}")
    if {r.get("dataset_version") for r in rows}!={"revised"} or {r.get("match_mode") for r in rows}!={"fuzzy"}:
        raise ValueError("Must use verified revised/fuzzy TRAIN source")
    if any("teacher_future" in r or "uav_pose" in r for r in rows):
        raise ValueError("Expected static-only TRAIN, not trajectory prefixes")
    catalog=json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    names=PublicNames(catalog)
    staged={mode:[] for mode in _VALID_MODES};audit=[];counts=Counter()
    for idx,row in enumerate(rows):
        text=row["instruction"];map_name=row["map_name"]
        if map_name not in catalog:
            raise ValueError(f"Public catalog missing map {map_name}")
        mentions=names.match(map_name,text)
        pairs=extract_pairs(text,mentions)
        refs,fail=public_references(map_name,row["world_bounds"],mentions,catalog,max_landmarks)
        # If invalid public geometry, NEVER pretend name-only text can bind
        # some different polygon; all arms get same explicit empty refs.
        if not refs:
            pairs=[]
        names_only=normalized_instruction(pairs,"landmark_only")
        compact=normalized_instruction(pairs,"landmark_relation")
        status=("no_public_anchor" if not refs else
                "paired_words" if any(p["phrase"] for p in pairs) else "name_only")
        if fail:
            counts.update(fail)
        counts[status]+=1
        audit.append(dict(index=idx,map_name=map_name,status=status,
            kept_pairs=[dict(name=p["name"],geometry_words=p["phrase"],
                             public_ids=p["public_ids"]) for p in pairs],
            public_polygon_count=len(refs),warnings=fail))
        for mode,instruction in (
            ("full_text",text),
            ("landmark_relation",compact),
            ("landmark_only",names_only),
        ):
            # This is the SAME row & target position in all arms. Only text
            # and independently public-catalog-matched reference Polygons
            # differ from original TRAIN. No target ID/rgb is copied into
            # neural inputs; original source is kept as an external label.
            record=copy.deepcopy(row)
            record["instruction"]=instruction
            record["landmarks"]=refs
            staged[mode].append(json.dumps(record,ensure_ascii=False))
    out_dir.mkdir(parents=True)
    files={}
    for mode,lines in staged.items():
        destination=out_dir/(mode+".jsonl")
        destination.write_text("\n".join(lines)+"\n",encoding="utf-8")
        files[mode]=dict(path=str(destination),sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                         records=len(lines))
    (out_dir/"extraction_audit.jsonl").write_text(
        "\n".join(json.dumps(x,ensure_ascii=False) for x in audit)+"\n",
        encoding="utf-8",
    )
    manifest=dict(
        protocol="unified-v2-public-landmark-geometry-phrase-paired-v1",
        source_sha256=hashlib.sha256(Path(source_path).read_bytes()).hexdigest(),
        catalog_sha256=hashlib.sha256(Path(catalog_path).read_bytes()).hexdigest(),
        all_queries=len(rows),counts=dict(counts),
        files=files,
        controls=list(_VALID_MODES),
        same_public_catalog_matched_polygon_for_all_arms=True,
        label_only_fields_unchanged=True,
        empty_text_uses_bert_unk_not_target_fallback=True,
        target_GT_used_to_select_polygon=False,
        parser_scope_accuracy_unverified=True,
        model_architecture_unchanged=True,
    )
    (out_dir/"manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n")
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl",type=Path,required=True)
    parser.add_argument("--public-objects",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--expected",type=int,default=21878)
    parser.add_argument("--max-landmarks",type=int,default=16)
    args=parser.parse_args()
    result=prepare(args.train_jsonl,args.public_objects,args.output,
                   expected=args.expected,max_landmarks=args.max_landmarks)
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
