"""Verify whether local 'revised' CityNav is ACTUALLY HETT refined data.

Naming a file 'revised' (or 'processed_citynav') is not evidence of
identity with the official HETT Refined CityNav release.

The caller supplies both official refined assets and current deployed assets.
This does NOT download data, modify labels, or read GT for model inference.

Example:
python -m scripts.audit_hett_refined_data \
    --current-val /DATA/sbfnav-data/processed_citynav/citynav_val_seen.json \
    --official-val /DOWNLOAD/HETT/refined/citynav_val_seen.json \
    --current-processed /DATA/sbfnav-data/cityrefer/processed_descriptions.json \
    --official-processed /DOWNLOAD/HETT/refined/processed_descriptions.json \
    --output /RUNS/hett_refined_provenance.json

The official paths above are illustrative. Verify their actual names and
serialization schema in the HETT authors' release rather than guessing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256(path):
    hasher=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def episodes(path):
    raw=json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw,list):
        raise ValueError(f"{path}: expected CityNav list of annotation episodes")
    rows={}
    for i,record in enumerate(raw):
        if not isinstance(record,dict) or not all(k in record for k in ("area","block","object_ids","ann_ids")):
            raise ValueError(f"{path}: unsupported CityNav episode schema at {i}")
        if not record["object_ids"] or not record["ann_ids"]:
            raise ValueError(f"{path}: episode {i} has missing object/annotation ID")
        key=(str(record["area"]).casefold(),str(record["block"]),
             str(record["object_ids"][0]),str(record["ann_ids"][0]))
        if key in rows:
            raise ValueError(f"{path}: duplicate CityNav episode identity {key}")
        desc=record.get("descriptions")
        if not isinstance(desc,list) or not desc:
            raise ValueError(f"{path}: no descriptions for episode {key}")
        rows[key]=dict(instruction=str(desc[0]),raw=record)
    return rows


def processed_keys(path):
    """Flatten source processed descriptions into map/object/annotation IDs."""
    raw=json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw,dict):
        raise ValueError(f"{path}: expected map->object->annotation structure")
    result={}
    for map_name,map_items in raw.items():
        if not isinstance(map_items,dict):
            raise ValueError("Invalid map dictionary")
        for object_id,annotations in map_items.items():
            if not isinstance(annotations,(list,dict)):
                raise ValueError("Invalid processed annotation dictionary")
            iterator=enumerate(annotations) if isinstance(annotations,list) else annotations.items()
            for ann_id,value in iterator:
                key=(str(map_name),str(object_id),str(ann_id))
                if key in result:raise ValueError("Duplicate processed-description annotation")
                result[key]=value
    return result


def compare_assets(current_path,official_path,*,kind,sample_limit=12):
    if not Path(current_path).is_file() or not Path(official_path).is_file():
        raise FileNotFoundError("Both current and OFFICIAL HETT refined files are required")
    current=episodes(current_path) if kind=="episodes" else processed_keys(current_path)
    official=episodes(official_path) if kind=="episodes" else processed_keys(official_path)
    paired=set(current)&set(official)
    only_local=set(current)-set(official)
    only_official=set(official)-set(current)
    if kind=="episodes":
        changed={k for k in paired if current[k]["instruction"]!=official[k]["instruction"]}
        full_changed={k for k in paired if current[k]["raw"]!=official[k]["raw"]}
    else:
        changed={k for k in paired if current[k]!=official[k]}
        full_changed=changed
    return dict(
        current_file=str(current_path),official_file=str(official_path),
        current_sha256=sha256(current_path),
        official_sha256=sha256(official_path),
        exact_byte_identical=(sha256(current_path)==sha256(official_path)),
        current_N=len(current),official_N=len(official),
        shared_ids=len(paired),
        only_current=len(only_local),only_official=len(only_official),
        changed_instructions_or_annotations=len(changed),
        changed_any_episode_fields=len(full_changed),
        semantic_identical=not(only_local or only_official or full_changed),
        example_only_current=[list(k) for k in sorted(only_local)[:sample_limit]],
        example_only_official=[list(k) for k in sorted(only_official)[:sample_limit]],
        example_changed=[list(k) for k in sorted(changed)[:sample_limit]],
    )


def audit(current_val,official_val,*,current_processed=None,
          official_processed=None):
    if bool(current_processed)!=bool(official_processed):
        raise ValueError("Compare both processed-description files or neither")
    val=compare_assets(current_val,official_val,kind="episodes")
    result=dict(
        purpose="CityNav local revised vs original authors HETT Refined release verification",
        val_seen=val,
        val_seen_exact_official_refined=val["exact_byte_identical"],
        val_seen_semantically_equivalent=val["semantic_identical"],
        current_revised_label_does_not_prove_official_refined=True,
        note="Refined CityNav landmark-annotation corrections are distinct "
             "from public Polygon shape validity and model capacity overflow. "
             "Val-Seen 2470 no-anchor counts from one dataset version cannot "
             "be transferred to a different version without re-audit.",
    )
    if current_processed:
        processed=compare_assets(current_processed,official_processed,kind="processed")
        result["processed_descriptions"]=processed
        result["all_supplied_semantically_equivalent"]=(
            val["semantic_identical"] and processed["semantic_identical"]
        )
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--current-val",type=Path,required=True)
    p.add_argument("--official-val",type=Path,required=True)
    p.add_argument("--current-processed",type=Path)
    p.add_argument("--official-processed",type=Path)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError("Refuse to overwrite provenance")
    result=audit(a.current_val,a.official_val,
                 current_processed=a.current_processed,
                 official_processed=a.official_processed)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",
                        encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
