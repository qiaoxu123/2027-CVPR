"""Frozen PGVB test: full named-landmark RGB vs real initial UAV RGB.

This is a PRIVILEGED, OFFLINE GOAL-LOCALIZATION DIAGNOSTIC, NOT navigation SR.
Each task is evaluated at its actual initial pose. The orthophoto crop is
derived solely from a uniquely matched public landmark Polygon at inference,
but its full RGB is unavailable to a causal UAV policy at time zero.

Three matched arms, SAME model and logits on each original task:
  static : Geometry + Relation, no RGB;
  current: single actual initial camera RGB, ungated PGVB;
  whole  : source-orthophoto RGB covering the ENTIRE named Polygon, ungated PGVB.

Both visual arms bypass the tracking gate equally to isolate *image content*.
Do not infer true RGB visibility from public Polygon footprint overlap.
GT target coordinates are read only after ALL three model inferences.
The test never changes weights, Stop, Goal thresholds, runtime or train split.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import hashlib
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

from graphnav.closedloop import PublicEpisode, PublicCatalog
from graphnav.checkpointing import file_sha256
from graphnav.privileged_landmark_rgb import privileged_whole_landmark_rgb
from graphnav.pgvb_observation import encode_landmark_observation
from graphnav.renderer import CityNavObservedViewRenderer
from graphnav.unified_full_relation_runtime import UnifiedPGVBFullRelationPolicy


def fingerprint(args):
    return dict(
        experiment="PRIVILEGED_full_landmark_rgb_offline_at_initial_pose",
        split=args.split,
        subset_max=args.max_episodes,
        data_root=str(args.data_root.resolve()),
        sbf_source=str(args.sbf_source.resolve()),
        config_sha=file_sha256(args.config),
        stage1_sha=file_sha256(args.checkpoint),
        original_unified_sha=file_sha256(args.unified_v2),
        trained_visual_relation_sha=file_sha256(args.relation_checkpoint),
        script_sha=file_sha256(__file__),
        crop_code_sha=file_sha256("graphnav/privileged_landmark_rgb.py"),
        relation_control="full",
        model_weights="frozen",
        visual_gate="OFF_FOR_BOTH_VISUAL_ARMS",
        source_rgb_privileged=True,
        valid_benchmark_closed_loop=False,
        GT_used_in_crop_or_input=False,
        context_margin_fraction=args.context_margin,
    )


def model_field(runtime, observed=None):
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        field = runtime.unified.forward_full_relation(
            static_inputs=runtime.inputs,
            tokens=runtime.tokens.float(),
            token_mask=runtime.mask,
            node_emb=runtime.names.float(),
            observed=observed,
            acquisition=None,  # matched ungated evidence, no fake repeated frames
            anchor_labels=None,
            relation_enabled=True,
        )["field"]
        scores = field.belief_logits.float()
        top = int(scores.argmax(-1)[0].item())
        point = runtime.world[0, top].detach().cpu().numpy().astype(float)
        return dict(
            top1_cell=top, xy=[float(point[0]), float(point[1])],
            margin_top2=float(scores.topk(2, -1).values.diff(dim=-1)[0, 0].abs()),
            max_logit=float(scores.max()),
        )


def score_posthoc(task_row, gt):
    """GT is accessible only AFTER every model_field call completes."""
    target = np.asarray(gt, dtype=float)[:2]
    for arm in ("static", "current", "whole"):
        prediction = task_row[arm]
        error = float(np.linalg.norm(np.asarray(prediction["xy"])-target))
        prediction.update(NE_m=error, R20=bool(error <= 20.0))
    return task_row


def summary(rows):
    if not rows:
        raise ValueError("No evaluation rows")
    eligible = [r for r in rows if r["anchor_status"] == "unique_supported"]
    paired = [r for r in eligible if r["whole_reason"] == "available"]
    def calc(group):
        if not group:
            return dict(N=0)
        out = dict(N=len(group))
        for name in ("static", "current", "whole"):
            out[name] = dict(
                R20=sum(bool(x[name]["R20"]) for x in group)/len(group),
                NE_m=sum(float(x[name]["NE_m"]) for x in group)/len(group),
            )
        for before, after in (("static","current"),("static","whole"),("current","whole")):
            out[before+"->"+after] = dict(
                corrected=sum(not x[before]["R20"] and x[after]["R20"] for x in group),
                damaged=sum(x[before]["R20"] and not x[after]["R20"] for x in group),
                changed_top1=sum(x[before]["top1_cell"] != x[after]["top1_cell"] for x in group),
            )
        return out
    return dict(
        label="PRIVILEGED SOURCE RGB, offline initial-Goal proxy, NOT navigation SR",
        full_dataset=calc(rows),
        unique_named_anchor=calc(eligible),
        whole_rgb_available=calc(paired),
        anchor_status=dict(Counter(r["anchor_status"] for r in rows)),
        current_reason=dict(Counter(r["current_reason"] for r in rows)),
        whole_reason=dict(Counter(r["whole_reason"] for r in rows)),
        sample_count=len(rows),
        ground_truth_posthoc_only=True,
        source_orthophoto_not_observed=True,
    )


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for item in ("sbf-source","config","data-root","checkpoint","unified-v2",
                 "relation-checkpoint","output"):
        p.add_argument("--"+item, type=Path, required=True)
    p.add_argument("--split", choices=("train_seen","val_seen"), default="train_seen")
    p.add_argument("--max-episodes", type=int, default=None,
                   help="Optional smoke ONLY; omit for full 2470 episodes")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--context-margin", type=float, default=0.25)
    p.add_argument("--save-paired-images", type=int, default=0,
                   help="Save at most this many local PNG pairs; DO NOT publish source images")
    a=p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Existing trained PGVB/SigLIP2 model requires CUDA")
    if a.max_episodes is not None and a.max_episodes < 1:
        raise ValueError("Invalid smoke sample size")
    if not (0 <= a.context_margin <= 1):
        raise ValueError("Invalid context margin")
    if a.output.exists() and not a.resume:
        raise FileExistsError("Refusing to overwrite a prior diagnostic")
    a.output.mkdir(parents=True,exist_ok=True)
    metadata=fingerprint(a)
    locked=a.output/"manifest.json"
    if a.resume:
        if not locked.exists() or json.loads(locked.read_text())!=metadata:
            raise ValueError("Cannot resume with different code/checkpoints/data")
    else:
        locked.write_text(json.dumps(metadata,indent=2))

    sys.path.insert(0,str(a.sbf_source.resolve()))
    from sbfnav.config import load_config,apply_overrides,validate_config
    from sbfnav.dataset import CityNavDataset,CityReferCatalog
    from sbfnav.planning_state import PlanningStateBuilder
    from sbfnav.mapdata import GROUND_LEVEL

    cfg=apply_overrides(load_config(a.config),["data.root="+str(a.data_root)])
    validate_config(cfg)
    dataset=CityNavDataset(str(a.data_root),a.split,
                           version=cfg["data"]["version"])
    catalog=CityReferCatalog(a.data_root/"cityrefer")
    state=cfg["state"]
    builder=PlanningStateBuilder(
        str(a.data_root),PublicCatalog(catalog),
        canvas_size=cfg["model"]["canvas_size"],
        field_size=cfg["model"]["field_size"],
        max_height_above_ground_m=state["max_height_above_ground_m"],
        trajectory_radius_m=state["trajectory_radius_m"],
    )
    renderer=CityNavObservedViewRenderer(
        a.data_root/"rgbd",a.output/"render_probes",image_size=384
    )
    runtime=UnifiedPGVBFullRelationPolicy(
        a.checkpoint,a.unified_v2,a.relation_checkpoint,
        a.output/"public_inputs",relation_control="full",
        visual_control="tracked",geometry_source="trained",
    )
    limit=min(len(dataset),a.max_episodes or len(dataset))
    rows_path=a.output/"initial_goal_pairs.jsonl"
    previous = []
    if a.resume and rows_path.exists():
        previous=[json.loads(z) for z in rows_path.read_text().splitlines() if z.strip()]
    existing={x["index"] for x in previous}
    if len(existing)!=len(previous) or any(i>=limit or i<0 for i in existing):
        raise ValueError("Resume task indices are inconsistent")
    saved=0
    with rows_path.open("a" if a.resume else "w") as file:
        for i in range(limit):
            if i in existing:
                continue
            e=dataset[i]
            public=PublicEpisode(map_name=e.map_name,
                                 instruction=e.instruction,
                                 start_pose=e.start_pose.copy(),
                                 references=tuple(catalog.referenced_landmarks(e)),
                                 reference_names=tuple(catalog.referenced_landmark_names(e)))
            runtime.set_episode(public,builder.transform(e.map_name),catalog)
            runtime._initialize_geometry_inputs()
            # Static field: no source map RGB access.
            static=model_field(runtime)
            current=static.copy()
            whole=static.copy()
            current_reason="not_applicable"
            whole_reason="not_applicable"
            details=None
            anchor_status=("unique_supported"
                           if runtime._landmark is not None else "no_unique_supported_visual_anchor")
            if runtime._landmark is not None:
                current_obs=renderer.current_rgbd(
                    e.map_name,e.start_pose,GROUND_LEVEL[e.map_name]
                )
                cur, current_reason, cur_meta = encode_landmark_observation(
                    current_obs,runtime._landmark,runtime.record,
                    runtime.native,runtime.device,return_metadata=True
                )
                if cur is not None:
                    current=model_field(runtime,cur)
                full_obs, details = privileged_whole_landmark_rgb(
                    renderer,e.map_name,runtime._landmark["contours"],
                    margin_fraction=a.context_margin,
                )
                full, whole_reason, full_meta = encode_landmark_observation(
                    full_obs,runtime._landmark,runtime.record,
                    runtime.native,runtime.device,return_metadata=True
                )
                if full is not None:
                    whole=model_field(runtime,full)
                details=dict(
                    **details,
                    current_polygon_whole_fraction=(
                        cur_meta.get("polygon_whole_fraction") if cur_meta else None),
                    current_source_support=(
                        cur_meta.get("source_support") if cur_meta else None),
                    full_polygon_whole_fraction=(
                        full_meta.get("polygon_whole_fraction") if full_meta else None),
                    full_source_support=(
                        full_meta.get("source_support") if full_meta else None),
                )
                if saved<a.save_paired_images:
                    examples=a.output/"local_privileged_rgb_png"
                    examples.mkdir(parents=True,exist_ok=True)
                    cv2.imwrite(str(examples/f"{i:05d}_current.png"),current_obs["bgr"])
                    cv2.imwrite(str(examples/f"{i:05d}_WHOLE_ORACLE.png"),full_obs["bgr"])
                    saved+=1
            row=dict(index=i,map_name=e.map_name,
                     anchor_status=anchor_status,
                     current_reason=current_reason,whole_reason=whole_reason,
                     source_crop=details,static=static,current=current,whole=whole)
            # Labels are never accessed until all inferences have finished.
            row=score_posthoc(row,e.target_position)
            file.write(json.dumps(row)+"\n");file.flush()
            previous.append(row)
            if (i+1)%100==0:
                print(json.dumps(dict(done=i+1,total=limit,
                                      eligible=sum(x["anchor_status"]=="unique_supported"
                                                   for x in previous))),flush=True)
    if len(previous)!=limit:
        raise RuntimeError(f"Coverage incomplete {len(previous)}/{limit}")
    result=dict(
        provenance=metadata,
        expected=len(dataset),
        observed=limit,
        smoke=bool(a.max_episodes),
        diagnostics=summary(previous),
    )
    (a.output/"summary.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    main()
