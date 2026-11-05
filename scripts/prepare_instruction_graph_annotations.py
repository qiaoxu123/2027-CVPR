"""Prepare HUMAN-REVIEW worklist from existing spaCy structured parser proposals.

Input: JSONL, one line per {"parsed": <structured_navigation.parse_instruction>}
(or the parsed dict itself). Output JSONL teacher proposals in the schema used
by scripts.train_instruction_relation_parser. EVERY line is reviewed=false.
Never convert these weak proposals into gold without explicit manual review.
"""
import argparse
import json
from pathlib import Path

from graphnav.instruction_relation_graph import (
    ROLES,PREDICATES,FRAMES,normalize_predicate,
)


def proposal_to_annotation(parsed):
    if parsed.get("target_GT_used") is not False:
        raise ValueError("Source must be public-text-only parse")
    if any(k in parsed for k in ("target_xy","target_world","target_position")):
        raise ValueError("Target oracle labels forbidden")
    entities=parsed.get("entities",[])
    target=(parsed.get("target") or {}).get("entity")
    if not target or target not in {e["id"] for e in entities}:
        return dict(instruction=parsed["original_text"],map_name=parsed["map_name"],
                    reviewed=False,spans=[],relations=[],
                    review_flags=["target_missing"]+parsed.get("ambiguity",[]))
    ids={e["id"]:i for i,e in enumerate(entities)}
    spans=[]
    for item in entities:
        role="target" if item["id"]==target else (
            "public_landmark" if item.get("role")=="landmark"
            else "intermediate" if item.get("role")=="reference_object"
            else "reference_object")
        spans.append(dict(start=int(item["span"][0]),end=int(item["span"][1]),
                          role=role,phrase=item.get("text",""),source_id=item["id"]))
    relations=[];flags=list(parsed.get("ambiguity",[]))
    for item in parsed.get("relations",[]):
        if item["subject"] not in ids or item["object"] not in ids:
            flags.append("relation_endpoint_missing")
            continue
        predicate=normalize_predicate(item.get("predicate",""))
        frame=(item.get("reference_frame") or {}).get("frame","unknown")
        if frame not in FRAMES:frame="unknown"
        if (predicate=="unknown" or item.get("relation_kind")!="spatial_proposal"
                or item.get("ambiguity")):
            flags.append("relation_needs_review")
        relations.append(dict(
            source=ids[item["subject"]],target=ids[item["object"]],
            predicate=predicate,frame=frame,
            source_predicate=item.get("predicate",""),
        ))
    return dict(instruction=parsed["original_text"],map_name=parsed["map_name"],
                reviewed=False,spans=spans,relations=relations,
                original_direct_paths=len(parsed.get("direct_relations",[])),
                original_indirect_paths=len(parsed.get("indirect_relations",[])),
                review_flags=sorted(set(flags)),
                target_GT_used=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parsed-jsonl",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError("Do not overwrite annotation worklist")
    if not args.output.parent.exists():args.output.parent.mkdir(parents=True)
    written=0
    with args.parsed_jsonl.open() as src,args.output.open("x") as dest:
        for line in src:
            if not line.strip():continue
            input_row=json.loads(line)
            parsed=input_row.get("parsed",input_row)
            row=proposal_to_annotation(parsed)
            dest.write(json.dumps(row,ensure_ascii=False)+"\n")
            written+=1
    if written==0:raise ValueError("No input parser proposals")
    print(json.dumps(dict(worklist=str(args.output),rows=written,
                          gold_labels_created=0,all_reviewed_false=True)))


if __name__=="__main__":main()
