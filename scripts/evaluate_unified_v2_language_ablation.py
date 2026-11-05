"""Paired, post-training TRAIN-Dev evaluation for original UnifiedPGVB-v2.

EXISTING fully trained full-text baseline is EVALUATED but NOT retrained.
Only two NEW text arms are independently trained from original Unified V2:
landmark+relation phrase and relation phrase only. All use the SAME public
Polygons and complete original query order. All source
questions remain in denominator.

No observed RGB is passed in this comparison; this isolates the main
Geometry+Relation language path before any causal visual branch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader,Subset

from graphnav.candidate_geometry import FullCandidateDataset,collate_full
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.full_relation_training import FORMAT,load_full_relation_model,full_relation_inputs
from graphnav.token_language import FrozenBERTTokens

ARM_NAMES=("full_text","landmark_relation","relation_only")


def load_data(path,conf,batch_size):
    data=FullCandidateDataset(
        path,grid_size=int(conf["grid_size"]),
        max_landmarks=int(conf["max_landmarks"]),
        max_roads=int(conf["max_roads"]),
        max_relations=int(conf["max_relations"]),
        prior_sigma_m=float(conf["prior_sigma_m"]),
        rule_sigma_m=float(conf["rule_sigma_m"]),
        deduplicate_queries=False)
    dev=[i for i,r in enumerate(data.records) if r["map_name"].lower().startswith("birmingham_")]
    if not dev:raise ValueError("Birmingham TRAIN-Dev has no tasks")
    if any(not(r["map_name"].lower().startswith("birmingham_") or
               r["map_name"].lower().startswith("cambridge_")) for r in data.records):
        raise ValueError("Unexpected map in full TRAIN")
    loader=DataLoader(Subset(data,dev),batch_size=batch_size,shuffle=False,
                      collate_fn=collate_full,num_workers=0)
    return data,dev,loader


def task_predictions(logits,batch):
    grids=batch["grid_xy"].to(logits.device).float()
    target=batch["target_xy"].to(logits.device).float()
    meters=batch["map_sizes"].to(logits.device).float()
    ids=logits.topk(15,-1).indices
    chosen=grids.gather(1,ids[:,:,None].expand(-1,-1,2))
    distances=((chosen-target[:,None])*meters[:,None]).norm(dim=-1)
    mass=torch.softmax(logits.float(),dim=-1)
    all_dist=((grids-target[:,None])*meters[:,None]).norm(dim=-1)
    within=(all_dist<=20.).float()
    return dict(
        top1_ne=distances[:,0].cpu(),
        r1=(distances[:,0]<=20.).cpu(),
        r5=(distances[:,:5].min(-1).values<=20.).cpu(),
        r15=(distances.min(-1).values<=20.).cpu(),
        mass20=(mass*within).sum(-1).cpu(),
        top1_id=ids[:,0].cpu(),
    )


@torch.no_grad()
def evaluate_arm(checkpoint,loader,device,batch_size):
    state=torch.load(checkpoint,map_location="cpu",weights_only=True)
    if state.get("format")!=FORMAT or state.get("completed_epochs",0)<1 or state.get("smoke"):
        raise ValueError("Require completed non-smoke Unified V2 full-relation checkpoint")
    model=load_full_relation_model(state,device).eval().requires_grad_(False)
    cfg=state["config"]["geometry_args"]
    bert=FrozenBERTTokens(cfg["text_backbone"],device=device,
                          max_length=int(cfg["max_text_length"]))
    data={k:[] for k in ("geometry_r1","relation_r1","r1","r5","r15",
                        "ne","mass20","id","geometry_ne","relation_ne")}
    for batch in loader:
        model.reset_episode() # IMPORTANT: frozen core caches previous batch if not reset.
        inputs=full_relation_inputs(batch,bert,device)
        result=model.forward_full_relation(**inputs,observed=None)
        geometry=task_predictions(result["geometry_field"].belief_logits,batch)
        relation=task_predictions(result["relation_field"].belief_logits,batch)
        for k,key in [("geometry_r1","r1"),("geometry_ne","top1_ne")]:
            data[k].append(geometry[key])
        for k,key in [("relation_r1","r1"),("relation_ne","top1_ne")]:
            data[k].append(relation[key])
        for k,key in [("r1","r1"),("r5","r5"),("r15","r15"),
                      ("ne","top1_ne"),("mass20","mass20"),
                      ("id","top1_id")]:
            data[k].append(relation[key])
    return {k:torch.cat(v) for k,v in data.items()}


def describe(result):
    n=len(result["r1"])
    return dict(N=n,R20=float(result["r1"].float().mean()),
                R5_20m=float(result["r5"].float().mean()),
                R15_20m=float(result["r15"].float().mean()),
                NE_m=float(result["ne"].float().mean()),
                mass20=float(result["mass20"].mean()),
                Geometry_R20=float(result["geometry_r1"].float().mean()),
                Geometry_NE_m=float(result["geometry_ne"].float().mean()),
                Relation_R20=float(result["relation_r1"].float().mean()),
                Relation_NE_m=float(result["relation_ne"].float().mean()))


def paired_stats(a,b):
    if len(a["r1"])!=len(b["r1"]):
        raise ValueError("Unmatched paired prediction lengths")
    return dict(
        fixes=int((~a["r1"]&b["r1"]).sum()),
        breaks=int((a["r1"]&~b["r1"]).sum()),
        net=int(b["r1"].sum()-a["r1"].sum()),
        changed_top1=int((a["id"]!=b["id"]).sum()),
        mean_NE_improvement_m=float((a["ne"]-b["ne"]).float().mean()),
    )


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared-dir",type=Path,required=True)
    p.add_argument("--existing-full-checkpoint",type=Path,required=True)
    p.add_argument("--checkpoint-landmark-relation",type=Path,required=True)
    p.add_argument("--checkpoint-relation-only",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--expected",type=int,default=21878)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError("Refuse to overwrite report")
    if not torch.cuda.is_available():raise RuntimeError("Evaluation requires CityNav GPU")
    if args.batch_size<1:raise ValueError("Invalid batch size")
    prepared=json.loads((args.prepared_dir/"manifest.json").read_text())
    if prepared["all_queries"]!=args.expected:raise ValueError("Unexpected prepared denominator")
    checkpoints=dict(
        full_text=args.existing_full_checkpoint,
        landmark_relation=args.checkpoint_landmark_relation,
        relation_only=args.checkpoint_relation_only)
    # Audit: equal map, normalized GT labels, and identical reference polygons
    # across all arm inputs. Target GT is used ONLY to compare after inference.
    records={};loaded={};dev_ids={}
    config=None
    for name in ARM_NAMES:
        state=torch.load(checkpoints[name],map_location="cpu",weights_only=True)
        if state.get("format")!=FORMAT or state.get("smoke"):
            raise ValueError("Invalid full training checkpoint")
        if name=="full_text":
            # Original model was trained ONCE before this experiment, using
            # the original source. Do NOT require a new full-text checkpoint.
            if prepared["source_sha256"]!=state["source_sha256"]["train_jsonl"]:
                raise ValueError("Existing baseline was not trained on original source queries")
        elif file_sha256(args.prepared_dir/(name+".jsonl")) != state["source_sha256"]["train_jsonl"]:
            raise ValueError(f"{name} was not trained on its prepared language variant")
        if state["completed_epochs"]!=state["planned_epochs"]:
            raise ValueError("Checkpoint training incomplete")
        conf=state["config"]["geometry_args"]
        if config is None:config=conf
        elif config!=conf:raise ValueError("Different Geometry configurations across arms")
        ds,ids,loader=load_data(args.prepared_dir/(name+".jsonl"),conf,args.batch_size)
        if len(ds)!=args.expected:raise ValueError("Incomplete source denominator")
        records[name]=ds.records
        loaded[name]=loader
        dev_ids[name]=ids
    for name in ARM_NAMES[1:]:
        if dev_ids[name]!=dev_ids["full_text"]:
            raise ValueError("Dev query order differs")
        for a,b in zip(records["full_text"],records[name]):
            for field in ("map_name","world_bounds","target_xy","physical_map_sizes","landmarks"):
                if a[field]!=b[field]:
                    raise ValueError(f"Paired input differs in non-language field {field}")
    outputs={}
    for name in ARM_NAMES:
        outputs[name]=evaluate_arm(checkpoints[name],loaded[name],"cuda",args.batch_size)
        print(json.dumps(dict(arm=name,summary=describe(outputs[name]))),flush=True)
    result=dict(
        protocol="unified-v2-trained-language-input-ablation-v1",
        source_manifest_sha=file_sha256(args.prepared_dir/"manifest.json"),
        checkpoints={n:file_sha256(p) for n,p in checkpoints.items()},
        all_train_queries=args.expected,
        birmingham_train_dev_queries=len(dev_ids["full_text"]),
        arms={n:describe(outputs[n]) for n in ARM_NAMES},
        paired={
            "existing_full_vs_landmark_relation":paired_stats(outputs["full_text"],outputs["landmark_relation"]),
            "existing_full_vs_relation_only":paired_stats(outputs["full_text"],outputs["relation_only"]),
            "relation_only_vs_landmark_relation":paired_stats(outputs["relation_only"],outputs["landmark_relation"]),
        },
        note=("Original full-text checkpoint reused without retraining; two newly trained compact-language variants. "
              "Same Unified V2 architecture, public Polygons, and frozen pretrained BERT. No RGB/Val/Test/closed-loop."),
        baseline_evaluation_input_shift=("Frozen original full-text checkpoint was trained with its original processed public reference binding; "
              "for paired comparison it is evaluated with newly catalog-matched polygons. Treat as input-shift baseline, "
              "not a retrained matched-distribution control."),
        warning="Spatial rule map is recomputed from respective text: a correct direction change may arise from deterministic rules, not necessarily learned language understanding.",
    )
    args.output.mkdir(parents=True)
    atomic_json(result,args.output/"summary.json")
    lines=[]
    for pos,idx in enumerate(dev_ids["full_text"]):
        row=dict(source_index=idx,
                 status="Birmingham TRAIN-Dev",
                 arms={n:dict(
                     hit20=bool(outputs[n]["r1"][pos]),
                     NE_m=float(outputs[n]["ne"][pos]),
                     top1_id=int(outputs[n]["id"][pos]))
                 for n in ARM_NAMES})
        lines.append(json.dumps(row))
    (args.output/"paired_tasks.jsonl").write_text("\n".join(lines)+"\n")
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    main()
