"""Offline causal replay and failure attribution for the frozen V2 Val-Seen run.

GT is read only after the public query has produced a frozen field and each
recorded pose has been replayed through the frozen trajectory/stop head.
RGB-D rendering is also post-hoc and uses only poses actually present in that
episode's logged rollout. No policy action or checkpoint is changed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch


def spatial_nms(order, xy, radius=20.0, k=20):
    """Stable score order plus fixed-radius spatial NMS, matching V2 recall."""
    kept = []
    for idx in order:
        if not kept or np.linalg.norm(xy[np.asarray(kept)] - xy[idx], axis=1).min() >= radius:
            kept.append(int(idx))
        if len(kept) == k:
            break
    return kept


def target_nms_rank(logits, xy, target, prefix, radius=20.0, k=20):
    """Score an already produced belief; target is strictly post-hoc."""
    prefix = [int(x) for x in prefix]
    seen = set(prefix)
    order = prefix + [int(x) for x in np.argsort(-logits, kind="stable") if int(x) not in seen]
    nms = spatial_nms(order, xy, radius=radius, k=k)
    distances = np.linalg.norm(xy - np.asarray(target, dtype=np.float64)[None], axis=1)
    rank = next((i + 1 for i, ix in enumerate(nms) if distances[ix] <= radius), None)
    eligible = np.flatnonzero(distances <= radius)
    top1 = int(order[0])
    return {
        "order": order,
        "nms": nms,
        "rank": rank,
        "top1_ne_m": float(distances[top1]),
        "top1_logit": float(logits[top1]),
        "best_target_logit": float(logits[eligible].max()) if len(eligible) else None,
        "score_gap_to_target20": float(logits[top1] - logits[eligible].max()) if len(eligible) else None,
        "nms_hits": {str(k): bool(any(distances[ix] <= radius for ix in nms[:k])) for k in (1, 5, 10, 20)},
        "nms_distances": [float(distances[ix]) for ix in nms],
        "nms_logits": [float(logits[ix]) for ix in nms],
        "nms_xy": xy[nms].astype(float).tolist(),
        "raw_top20_xy": xy[order[:20]].astype(float).tolist(),
        "raw_top20_cell": [int(x) for x in order[:20]],
        "all_cell_rank": int(np.flatnonzero(np.argsort(-logits, kind="stable") == top1)[0]) + 1,
    }


def normalize_category(object_type):
    value = re.sub(r"[^a-z]+", " ", str(object_type or "").lower()).strip()
    if any(x in value.split() for x in ("car", "vehicle", "van", "truck", "bus")):
        return "Car"
    if "building" in value or "house" in value or "structure" in value or "church" in value:
        return "Building"
    if "parking" in value or "carpark" in value or "car park" in value:
        return "Parking"
    if any(x in value.split() for x in ("ground", "grass", "land", "field", "open", "space", "garden")):
        return "Ground"
    return "Other"


def relation_proxy(instruction, ref_names):
    """Conservative language complexity proxy; not a verified relation parse."""
    text = instruction.lower()
    if not ref_names:
        return "unbound_or_unmentioned"
    if any(x in text for x in ("between", "row of", "surrounded by", "surrounding", "enclosed by", "inside the parking", "in the parking lot", "in a parking lot", "parking area")):
        return "indirect_or_multi-entity_proxy"
    direct = re.compile(r"\b(north|south|east|west|near|next to|beside|behind|in front of|across from|opposite|inside|outside|within|along|on|at|left|right)\b", re.I)
    return "direct_relation_word_proxy" if direct.search(text) else "relation_uncertain"


def primary_failure(sr, osr, nms_rank, selected_correct_any, target_contour_ok=True):
    if sr:
        return "success"
    if osr:
        return "entered_20m_then_left"
    if nms_rank is None:
        return "no_correct_candidate_in_spatial_nms20"
    if nms_rank <= 5 and not selected_correct_any:
        return "nms_top5_contains_target_but_policy_never_selected_it"
    if nms_rank <= 5:
        return "nms_top5_contains_target_but_trajectory_did_not_reach_it"
    if nms_rank <= 20:
        return "correct_candidate_only_at_nms_ranks6_20"
    return "other_or_unscored"


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    tmp.replace(path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def target_object(catalog, map_name, object_id):
    items = catalog.objects[map_name]
    if isinstance(items, dict):
        items = items.values()
    for item in items:
        if int(item.get("id", -1)) == int(object_id):
            return item
    return None


def _polygon_mask(contour_world, observation):
    from graphnav.landmark_projection import world_to_frame
    bgr = observation["bgr"]
    h, w = bgr.shape[:2]
    contour = np.asarray(contour_world, dtype=np.float32)
    if contour.ndim != 2 or contour.shape[1] != 2 or len(contour) < 3 or not np.isfinite(contour).all():
        return None
    try:
        px = world_to_frame(contour, observation["footprint_world"], bgr.shape)
    except (ValueError, cv2.error):
        return None
    if not np.isfinite(px).all() or np.max(np.abs(px)) > 1e6:
        return None
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(px).astype(np.int32)], 1)
    ys, xs = np.where(mask != 0)
    if len(xs) == 0:
        return {"mask": mask, "bbox": None, "long_px": 0.0, "short_px": 0.0, "pixels": 0}
    return {"mask": mask, "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "long_px": float(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)),
            "short_px": float(min(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)),
            "pixels": int(mask.sum()), "contour_px": px.tolist()}


def observe_actual_history(ds, catalog, rows, rgbd_dir, output_dir, ground_levels):
    """Post-hoc map/pixel-support proxy on logged poses only; never model input."""
    from graphnav.renderer import CityNavObservedViewRenderer
    renderer = CityNavObservedViewRenderer(rgbd_dir, output_dir, image_size=384)
    summaries = []
    gallery_frames = {}
    for i, row in enumerate(rows):
        record = ds[i]  # post-hoc diagnostics only
        obj = target_object(catalog, record.map_name, record.object_id)
        contour = None if obj is None else obj.get("contour")
        observations = []
        best = None
        poses = np.asarray(row["poses"], dtype=np.float64)
        for t, pose in enumerate(poses):
            observation = renderer.current_rgbd(record.map_name, pose, ground_levels[record.map_name])
            projected = _polygon_mask(contour, observation) if contour is not None else None
            if projected is None:
                metric = {"step": t, "geometry_support": False, "max_dim_px": 0.0,
                          "min_dim_px": 0.0, "bbox_pixels": 0, "depth_valid_fraction": None}
            else:
                mask = projected["mask"].astype(bool)
                visible_pixels = int(mask.sum())
                valid_pixels = int((mask & observation["depth_valid"]).sum())
                rgb_pixels = int((mask & (observation["bgr"].max(-1) > 0)).sum())
                metric = {"step": t, "geometry_support": visible_pixels > 0,
                          "max_dim_px": projected["long_px"], "min_dim_px": projected["short_px"],
                          "bbox_pixels": visible_pixels, "rgb_nonblack_pixels": rgb_pixels,
                          "depth_valid_fraction": valid_pixels / visible_pixels if visible_pixels else 0.0,
                          "footprint_world": observation["footprint_world"].tolist(),
                          "target_contour_px": projected.get("contour_px")}
                score = (projected["long_px"] >= 32 and projected["short_px"] >= 16,
                         projected["pixels"], projected["long_px"])
                if best is None or score > best[0]:
                    best = (score, t, observation["bgr"].copy(), metric)
            observations.append(metric)
        first_support = next((x["step"] for x in observations if x["geometry_support"]), None)
        first_16 = next((x["step"] for x in observations if x["max_dim_px"] >= 16), None)
        first_32x16 = next((x["step"] for x in observations if x["max_dim_px"] >= 32 and x["min_dim_px"] >= 16), None)
        summaries.append({
            "index": i, "target_object_available": obj is not None,
            "target_object_type": None if obj is None else obj.get("object_type"),
            "target_contour_available": contour is not None,
            "actual_logged_pose_count": len(poses),
            "first_geometric_projection_step": first_support,
            "first_long_side_16px_step": first_16,
            "first_long32_short16_proxy_step": first_32x16,
            "remaining_actions_after_proxy": None if first_32x16 is None else max(0, len(poses)-1-first_32x16),
            "ever_long_side_16px": first_16 is not None,
            "ever_long32_short16_proxy": first_32x16 is not None,
            "max_long_side_px": max((x["max_dim_px"] for x in observations), default=0.0),
            "max_short_side_px": max((x["min_dim_px"] for x in observations), default=0.0),
            "any_depth_valid_target_pixels": any((x.get("depth_valid_fraction") or 0) > 0 for x in observations),
            "best_proxy_frame_step": None if best is None else best[1],
            "same_type_co_visible_objects_best_proxy_frame": None,
            "similar_scale_same_type_objects_best_proxy_frame": None,
            "frames": observations,
        })
        if best is not None:
            from graphnav.landmark_projection import world_to_frame
            frame = {"step": best[1], "bgr": best[2], "metric": best[3]}
            gallery_frames[i] = frame
            footprint = np.asarray(best[3]["footprint_world"], dtype=np.float32)
            target_type = None if obj is None else obj.get("object_type")
            target_dims = np.asarray((obj or {}).get("dimension", [0,0,0]), dtype=float)[:2]
            target_size_m = max(float(target_dims.max()) if target_dims.size else 0.0, 1e-3)
            target_scale_px = max(float(best[3]["max_dim_px"]), 1e-3)
            same_type, similar_scale = 0, 0
            if target_type is not None:
                for other in catalog.objects[record.map_name].values():
                    if int(other.get("id", -1)) == record.object_id or other.get("object_type") != target_type:
                        continue
                    contour_other = other.get("contour")
                    if not contour_other or len(contour_other) < 3:
                        continue
                    try:
                        other_px = world_to_frame(np.asarray(contour_other, np.float32), footprint, (384,384,3))
                    except (ValueError, cv2.error):
                        continue
                    if not np.isfinite(other_px).all() or np.max(np.abs(other_px)) > 1e6:
                        continue
                    clipped = np.column_stack((np.clip(other_px[:,0],0,383),np.clip(other_px[:,1],0,383)))
                    other_mask = np.zeros((384,384), np.uint8)
                    cv2.fillPoly(other_mask,[np.rint(clipped).astype(np.int32)],1)
                    if not other_mask.any():
                        continue
                    same_type += 1
                    ys,xs=np.where(other_mask)
                    scale=max(xs.max()-xs.min()+1,ys.max()-ys.min()+1)
                    if .5 <= scale/target_scale_px <= 2.0:
                        similar_scale += 1
            summaries[-1]["same_type_co_visible_objects_best_proxy_frame"] = same_type
            summaries[-1]["similar_scale_same_type_objects_best_proxy_frame"] = similar_scale
        if (i + 1) % 250 == 0:
            print(f"observed-history posthoc {i+1}/2470", flush=True)
    return summaries, gallery_frames


def draw_case(case_id, legacy_q, expanded_q, old_field, new_field, old_row, row, target_obj,
              target_xy, rgb_frame, image_path, map_size=28):
    from io import BytesIO
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Matplotlib's system font cache does not always index Noto's TTC face by
    # the locale alias. Register the installed CJK font explicitly.
    from matplotlib import font_manager
    cjk_font = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if Path(cjk_font).is_file():
        font_manager.fontManager.addfont(cjk_font)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=cjk_font).get_name()
    else:
        matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    from matplotlib.patches import Polygon as MplPolygon
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), constrained_layout=True)
    poses = np.asarray(row["poses"], float)
    bounds = np.asarray(expanded_q["world_bounds"], float)
    x0, y0, x1, y1 = bounds
    for ax, q, field, navrow, title, path_color, goal_color in (
            (axes[0], legacy_q, old_field, old_row, "旧名称绑定 Belief", "#ffb15c", "#ff9a24"),
            (axes[1], expanded_q, new_field, row, "扩展名称绑定 Belief", "white", "#ffdf43")):
        z = np.asarray(field["logits"], np.float32).reshape(map_size, map_size)
        ax.imshow(z, origin="lower", extent=[x0, x1, y0, y1], cmap="magma", alpha=.85, aspect="auto")
        for lm in q.get("landmarks", []):
            c = np.asarray(lm.get("contour", []), float)
            if c.ndim == 2 and len(c) >= 3:
                world = np.column_stack((x0 + c[:, 0]*(x1-x0), y0 + c[:, 1]*(y1-y0)))
                ax.add_patch(MplPolygon(world, closed=True, fill=False, edgecolor="#38d9ff", linewidth=1.0, alpha=.9))
        if target_obj and target_obj.get("contour"):
            ax.add_patch(MplPolygon(np.asarray(target_obj["contour"], float), closed=True, fill=False,
                                    edgecolor="#57ff69", linewidth=1.5, label="GT Polygon（事后）"))
        ax.plot(target_xy[0], target_xy[1], marker="*", color="#57ff69", markersize=12, label="GT中心（事后）")
        navposes=np.asarray(navrow["poses"],float)
        ax.plot(navposes[:, 0], navposes[:, 1], color=path_color, linewidth=1.0, alpha=.85, label="实际轨迹")
        ax.scatter([navposes[0,0]], [navposes[0,1]], c="#43f3ff", marker="o", s=32, label="起点")
        ax.scatter([navposes[-1,0]], [navposes[-1,1]], c="#ff4444", marker="x", s=50, label="终点")
        if navrow.get("step_diagnostics"):
            goal = navrow["step_diagnostics"][-1].get("selected_goal_xy")
            if goal:
                ax.scatter([goal[0]], [goal[1]], c=goal_color, marker="D", s=30, label="最终选定 Goal")
        ax.set_title(f"{title}\n公开地标：{', '.join(l['name'] for l in q.get('landmarks', [])) or '无'}")
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal", adjustable="box")
        ax.legend(fontsize=7, loc="best")
    if rgb_frame is not None:
        rgb = cv2.cvtColor(rgb_frame["bgr"], cv2.COLOR_BGR2RGB)
        axes[2].imshow(rgb)
        contour = rgb_frame["metric"].get("target_contour_px")
        if contour:
            c = np.asarray(contour, float)
            axes[2].plot(np.r_[c[:,0], c[0,0]], np.r_[c[:,1], c[0,1]], color="#57ff69", lw=1.5)
        axes[2].set_title(f"真实已访问观测代理帧 step={rgb_frame['step']}\nGT Polygon投影（仅诊断，非分割真值）")
    else:
        axes[2].text(.5,.5,"没有可投影目标轮廓/历史帧",ha="center",va="center")
    axes[2].axis("off")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    buffer = BytesIO()
    fig.savefig(buffer, format="jpeg", dpi=100)
    buffer.seek(0)
    Image.open(buffer).convert("RGB").save(image_path, quality=88, optimize=True)
    plt.close(fig)


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(""); return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict,list)) else v for k,v in row.items()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "data", "sbf-source", "config", "source-v2", "trained-v2", "stage1",
                 "legacy-queries", "expanded-queries", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--skip-observation-replay", action="store_true")
    args = p.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    sys.path.insert(0, str(args.sbf_source.resolve()))
    from sbfnav.config import load_config, apply_overrides
    from sbfnav.dataset import CityNavDataset, CityReferCatalog
    from sbfnav.mapdata import GROUND_LEVEL
    from graphnav.checkpointing import file_sha256
    from graphnav.closedloop import PublicCatalog
    from graphnav.dynamic_runtime import NativeObservationBuilder
    from scripts.evaluate_v2_expanded_public_navigation import FrozenV2GeometryPolicy

    torch.manual_seed(42); np.random.seed(42); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("This diagnostic must replay the frozen GPU runtime; no CPU substitution")
    ds = CityNavDataset(args.data, "val_seen", version="processed_citynav")
    catalog = CityReferCatalog(args.data / "cityrefer")
    legacy_q, expanded_q = read_jsonl(args.legacy_queries), read_jsonl(args.expanded_queries)
    if len(ds) != 2470 or len(legacy_q) != 2470 or len(expanded_q) != 2470:
        raise ValueError("Expected exactly 2,470 paired Val-Seen records and queries")
    run_metrics = json.loads((args.run / "metrics.json").read_text())
    if run_metrics["coverage_per_arm"] != {"legacy": 2470, "expanded": 2470}:
        raise ValueError("Prior complete closed-loop coverage mismatch")
    old_rows = [json.loads((args.run/"legacy"/"rows"/f"{i:05d}.json").read_text()) for i in range(2470)]
    new_rows = [json.loads((args.run/"expanded"/"rows"/f"{i:05d}.json").read_text()) for i in range(2470)]
    for i, (a, b, qa, qb) in enumerate(zip(old_rows, new_rows, legacy_q, expanded_q)):
        if a["index"] != i or b["index"] != i or a["episode_id"] != b["episode_id"]:
            raise ValueError(f"Episode pair mismatch at {i}")
        record = ds[i]
        if list(record.episode_id) != a["episode_id"]:
            raise ValueError(f"Dataset identity mismatch at {i}")
        for q in (qa, qb):
            if q.get("target_xy") != [0.0, 0.0] or q["instruction"] != record.instruction or q["map_name"] != record.map_name:
                raise ValueError(f"Public-only query mismatch at {i}")

    out = args.output; out.mkdir(parents=True, exist_ok=True)
    fields_dir = out / "expanded_field_cache"; fields_dir.mkdir(exist_ok=True)
    config = apply_overrides(load_config(args.config), ["data.root=" + str(args.data)])
    builder_base = None
    policy = FrozenV2GeometryPolicy(args.source_v2, args.trained_v2, args.stage1,
                                    out / "scratch_public_inputs", "trained_v2")
    metrics_rows = []
    per_step_rows = []
    started = time.perf_counter()
    for i, (q, rollout_row) in enumerate(zip(expanded_q, new_rows)):
        policy.record = q
        policy.visited_goal_cells = set()
        field_path = fields_dir / f"{i:05d}.npz"
        if field_path.is_file():
            payload = np.load(field_path)
            logits = payload["logits"].astype(np.float32)
            xy = payload["world_xy"].astype(np.float64)
            prefix = payload["candidate_ids"].astype(int).tolist()
        else:
            field, inputs = policy.frozen_field()
            logits = field.belief_logits[0].float().cpu().numpy().astype(np.float32)
            bounds = np.asarray(q["world_bounds"], dtype=np.float64)
            grid = inputs["grid_xy"][0].float().cpu().numpy().astype(np.float64)
            xy = bounds[:2] + grid * (bounds[2:] - bounds[:2])
            prefix = field.candidate_ids[0].cpu().numpy().astype(int).tolist()
            tmp = field_path.with_suffix(".npz.tmp")
            with tmp.open("wb") as f:
                np.savez_compressed(f, logits=logits, world_xy=xy, candidate_ids=np.asarray(prefix, np.int32))
            tmp.replace(field_path)
        record = ds[i]  # target GT is intentionally accessed only after the model forward above.
        target = record.target_position[:2]
        ranking = target_nms_rank(logits, xy, target, prefix, radius=20.0, k=20)
        selected_candidates, selected_goals, stop_probs, stop_guard_fail = [], [], [], []
        saved_diag = rollout_row["step_diagnostics"]
        poses = np.asarray(rollout_row["poses"], dtype=np.float64)
        if len(saved_diag) != len(poses):
            raise ValueError(f"Logged pose/decision count mismatch at {i}: {len(poses)} vs {len(saved_diag)}")
        field, _ = policy.frozen_field()
        policy.visited_goal_cells = set()
        for t, (pose, old_diag) in enumerate(zip(poses, saved_diag)):
            prediction = policy.predict_field(SimpleNamespace(pose=pose), field)
            diag = prediction.diagnostics
            candidate_slot = int(diag["candidate"])
            grid_id = int(prefix[candidate_slot])
            candidate_xy = xy[grid_id]
            goal_xy = np.asarray(diag["selected_goal_xy"], dtype=np.float64)
            goal_dist = float(np.linalg.norm(goal_xy-target))
            selected_candidates.append({"slot": candidate_slot, "grid_id": grid_id, "distance_to_target_m": float(np.linalg.norm(candidate_xy-target))})
            selected_goals.append(goal_xy)
            stop_probs.append(float(diag["stop_probability"]))
            guard_failed = bool(float(diag["predicted_goal_distance"] or 0.0) > 1e-3 or not diag["stop_altitude_guard"])
            stop_guard_fail.append(guard_failed)
            # Verify the frozen controller can be reconstructed from recorded causal poses.
            for key in ("candidate", "mode"):
                if int(diag[key]) != int(old_diag[key]):
                    raise RuntimeError(f"Frozen controller replay mismatch task={i}, step={t}, field={key}")
            if not np.allclose(goal_xy, old_diag["selected_goal_xy"], atol=2e-3, rtol=0):
                raise RuntimeError(f"Frozen Goal replay mismatch task={i}, step={t}")
            if abs(float(diag["stop_probability"])-float(old_diag["stop_probability"])) > 2e-3:
                raise RuntimeError(f"Frozen Stop replay mismatch task={i}, step={t}")
            pose_dist = float(np.linalg.norm(pose[:2]-target))
            per_step_rows.append({
                "index": i, "step": t, "map_name": record.map_name,
                "pose_x": float(pose[0]), "pose_y": float(pose[1]), "pose_alt_m": float(pose[2]), "yaw_rad": float(pose[3]),
                "pose_gt_distance_m_posthoc": pose_dist,
                "belief_nms_rank_of_gt20_static": ranking["rank"],
                "belief_top1_ne_m_static": ranking["top1_ne_m"],
                "selected_candidate_slot": candidate_slot, "selected_grid_id": grid_id,
                "selected_candidate_gt_distance_m_posthoc": float(np.linalg.norm(candidate_xy-target)),
                "selected_goal_x": float(goal_xy[0]), "selected_goal_y": float(goal_xy[1]),
                "selected_goal_gt_distance_m_posthoc": goal_dist,
                "stop_probability": float(diag["stop_probability"]),
                "predicted_goal_distance_m": float(diag["predicted_goal_distance"] or 0.0),
                "stop_altitude_guard": bool(diag["stop_altitude_guard"]),
                "stop_would_trigger": bool(prediction.should_stop),
            })
        selected_goal_dist = [float(np.linalg.norm(x-target)) for x in selected_goals]
        goal_jumps = [float(np.linalg.norm(b-a)) for a,b in zip(selected_goals[:-1],selected_goals[1:])]
        path = np.asarray(rollout_row["poses"], dtype=np.float64)
        path_dist = np.linalg.norm(path[:,:2]-target[None], axis=1)
        osr = bool(path_dist.min() <= 20.0); sr = bool(rollout_row["metrics"]["SR"])
        rank = ranking["rank"]
        selected_correct_any = any(x["distance_to_target_m"] <= 20.0 for x in selected_candidates)
        primary = primary_failure(sr, osr, rank, selected_correct_any)
        query_anchor_names = [x["name"] for x in q.get("landmarks", [])]
        anchor_freq = Counter(query_anchor_names)
        bound_polygons = len(query_anchor_names)
        old_names = old_rows[i].get("public_reference_names", [])
        new_names = new_rows[i].get("public_reference_names", [])
        old_q, new_q = legacy_q[i], expanded_q[i]
        query_change = (old_q.get("landmarks", []) != new_q.get("landmarks", []))
        obj = target_object(catalog, record.map_name, record.object_id)
        category = normalize_category(None if obj is None else obj.get("object_type"))
        dist_to_selected = selected_goal_dist[-1] if selected_goal_dist else None
        last_diag = saved_diag[-1] if saved_diag else {}
        stop_threshold = .3
        stop_anomaly = bool(rollout_row["termination"] == "policy_stop" and
                            (not stop_probs[-1] >= stop_threshold or stop_guard_fail[-1]))
        metrics_rows.append({
            "index": i, "episode_id": json.dumps(record.episode_id), "map_name": record.map_name,
            "instruction": record.instruction, "target_object_id": record.object_id,
            "target_category_proxy": category, "target_object_type": None if obj is None else obj.get("object_type"),
            "expanded_anchor_names": new_names, "anchor_polygon_count": bound_polygons,
            "anchor_name_polygon_multiplicity": {k:v for k,v in anchor_freq.items() if v > 1},
            "public_anchor_count": len(set(new_names)),
            "legacy_anchor_names": old_names,
            "legacy_to_expanded_added_names": sorted(set(new_names)-set(old_names)),
            "legacy_to_expanded_removed_names": sorted(set(old_names)-set(new_names)),
            "landmark_query_geometry_changed": query_change,
            "relation_complexity_proxy": relation_proxy(record.instruction, new_names),
            "sr": int(sr), "osr": int(osr), "entered_then_left": int(osr and not sr),
            "min_pose_gt_distance_m_posthoc": float(path_dist.min()),
            "final_ne_m": float(path_dist[-1]),
            "initial_top1_ne_m": ranking["top1_ne_m"],
            "initial_nms_gt_rank20m": rank,
            "initial_nms_recall_at_1": int(ranking["nms_hits"]["1"]),
            "initial_nms_recall_at_5": int(ranking["nms_hits"]["5"]),
            "initial_nms_recall_at_10": int(ranking["nms_hits"]["10"]),
            "initial_nms_recall_at_20": int(ranking["nms_hits"]["20"]),
            "score_gap_top1_to_best_gt20_logit": ranking["score_gap_to_target20"],
            "selected_target_candidate_ever_within20": int(selected_correct_any),
            "selected_goal_never_within20": int(not selected_correct_any),
            "selected_goal_first_near_target_step": next((t for t,x in enumerate(selected_candidates) if x["distance_to_target_m"]<=20), None),
            "selected_goal_hop_count_over20m": sum(x>20 for x in goal_jumps),
            "selected_goal_max_hop_m": max(goal_jumps, default=0.0),
            "selected_goal_final_ne_m": dist_to_selected,
            "stop_probability_final": stop_probs[-1] if stop_probs else None,
            "stop_guard_anomaly": int(stop_anomaly),
            "termination": rollout_row["termination"],
            "primary_failure_class": primary,
            "tags": [tag for tag,yes in (
                ("no_public_anchor", not new_names),
                ("multiple_polygons_same_name", any(v>1 for v in anchor_freq.values())),
                ("multiple_public_reference_names", len(set(new_names))>1),
                ("binding_changed_vs_legacy", query_change),
                ("goal_hop_over_20m", any(x>20 for x in goal_jumps)),
                ("stopped_at_goal_over_20m_from_gt", bool(rollout_row["termination"]=="policy_stop" and dist_to_selected is not None and dist_to_selected>20)),
            ) if yes],
            "step_count": len(path)-1, "same_pose_trace_vs_legacy": int(path.tolist()==old_rows[i]["poses"]),
        })
        if (i+1) % 100 == 0:
            print(f"replayed geometry/controller {i+1}/2470 elapsed={time.perf_counter()-started:.1f}s", flush=True)

    del policy
    torch.cuda.empty_cache()
    atomic_json({"episodes": metrics_rows}, out/"episode_attribution.json")
    atomic_json({"steps": per_step_rows}, out/"per_step_replay.json")
    write_csv(out/"per_episode_attribution.csv", metrics_rows)
    write_csv(out/"per_step_replay.csv", per_step_rows)

    # Actual observed RGB-D support is a post-hoc projection proxy only.
    if args.skip_observation_replay:
        obs_summaries, gallery_frames = [], {}
    else:
        obs_summaries, gallery_frames = observe_actual_history(
            ds, catalog, new_rows, args.data/"rgbd", out/"unused_render_dir", GROUND_LEVEL)
    atomic_json({"episodes": obs_summaries}, out/"historical_rgbd_projection_proxy.json")
    write_csv(out/"historical_rgbd_projection_proxy.csv", [{k:v for k,v in x.items() if k!="frames"} for x in obs_summaries])

    # Binding correction/damage cases and representative failure modes.
    corrected = [r["index"] for r in metrics_rows if old_rows[r["index"]]["metrics"]["SR"] == 0 and new_rows[r["index"]]["metrics"]["SR"] == 1]
    damaged = [r["index"] for r in metrics_rows if old_rows[r["index"]]["metrics"]["SR"] == 1 and new_rows[r["index"]]["metrics"]["SR"] == 0]
    counts = Counter(r["primary_failure_class"] for r in metrics_rows)
    gallery_ids = set(corrected + damaged)
    for cause in sorted(counts):
        if cause == "success":
            continue
        examples = [r["index"] for r in metrics_rows if r["primary_failure_class"]==cause and r["index"] not in gallery_ids][:5]
        gallery_ids.update(examples)
        if len(gallery_ids) > 120:
            break
    gallery_ids = sorted(gallery_ids)
    print(f"cases: corrected={len(corrected)} damaged={len(damaged)} gallery={len(gallery_ids)}", flush=True)

    # Generate fields for binding-delta cases only; no labels enter model input.
    policy = FrozenV2GeometryPolicy(args.source_v2,args.trained_v2,args.stage1,out/"scratch_case_inputs","trained_v2")
    old_field_cache, new_field_cache = {}, {}
    for idx in gallery_ids:
        fields = {}
        for label,q in (("old",legacy_q[idx]),("new",expanded_q[idx])):
            policy.record=q; policy.visited_goal_cells=set()
            field,inputs=policy.frozen_field()
            z=field.belief_logits[0].float().cpu().numpy().astype(np.float32)
            b=np.asarray(q["world_bounds"],float); grid=inputs["grid_xy"][0].float().cpu().numpy()
            xy=b[:2]+grid*(b[2:]-b[:2])
            fields[label]={"logits":z,"world_xy":xy}
        record=ds[idx]; target_obj=target_object(catalog,record.map_name,record.object_id)
        render=gallery_frames.get(idx)
        img_path=out/"cases"/f"case_{idx:04d}.jpg"
        draw_case(idx,legacy_q[idx],expanded_q[idx],fields["old"],fields["new"],old_rows[idx],new_rows[idx],target_obj,record.target_position[:2],render,img_path)
    del policy
    torch.cuda.empty_cache()

    # Write all 74 paired binding outcomes, with explicit public-input deltas.
    delta_rows=[]
    for idx in corrected+damaged:
        r=metrics_rows[idx]
        delta_rows.append({k:r[k] for k in (
            "index","episode_id","map_name","instruction","target_object_id","target_category_proxy",
            "legacy_anchor_names","expanded_anchor_names","legacy_to_expanded_added_names",
            "legacy_to_expanded_removed_names","landmark_query_geometry_changed",
            "initial_nms_gt_rank20m","initial_top1_ne_m","final_ne_m","sr","osr",
            "primary_failure_class","tags")})
        delta_rows[-1]["legacy_sr"]=old_rows[idx]["metrics"]["SR"]
        delta_rows[-1]["expanded_sr"]=new_rows[idx]["metrics"]["SR"]
        delta_rows[-1]["case_image"]=f"cases/case_{idx:04d}.png"
    write_csv(out/"public_binding_corrected_damaged_74.csv",delta_rows)

    # Aggregate exact quantitative diagnostics.
    failed = [r for r in metrics_rows if not r["sr"]]
    never = [r for r in failed if not r["osr"]]
    summary = {
        "complete": len(metrics_rows)==2470,
        "n": len(metrics_rows),
        "coverage": {"legacy":len(old_rows),"expanded":len(new_rows)},
        "source_run_commit": "4e0758e",
        "primary_failure_counts_all": dict(counts),
        "primary_failure_counts_failed_only": dict(Counter(r["primary_failure_class"] for r in failed)),
        "multilabel_counts": {tag:sum(tag in r["tags"] for r in metrics_rows) for tag in sorted(set(t for r in metrics_rows for t in r["tags"]))},
        "never_entered_20m": len(never),
        "reached_then_left": sum(r["entered_then_left"] for r in metrics_rows),
        "never_entered_topk_recall": {str(k):sum(r[f"initial_nms_recall_at_{k}"] for r in never)/len(never) if never else None for k in (1,5,10,20)},
        "never_entered_topk_counts": {str(k):sum(r[f"initial_nms_recall_at_{k}"] for r in never) for k in (1,5,10,20)},
        "never_entered_top5_contains_gt_but_policy_not_select": sum(r["initial_nms_gt_rank20m"] is not None and r["initial_nms_gt_rank20m"]<=5 and not r["selected_target_candidate_ever_within20"] for r in never),
        "never_entered_top5_contains_gt_and_selected_but_route_missed": sum(r["initial_nms_gt_rank20m"] is not None and r["initial_nms_gt_rank20m"]<=5 and r["selected_target_candidate_ever_within20"] for r in never),
        "gt_nms_recall_all": {str(k):sum(r[f"initial_nms_recall_at_{k}"] for r in metrics_rows)/2470 for k in (1,5,10,20)},
        "final_failures_stop_guard_anomalies": sum(r["stop_guard_anomaly"] for r in failed),
        "all_episode_goal_hop_over20": sum(r["selected_goal_hop_count_over20m"]>0 for r in metrics_rows),
        "mean_selected_goal_hops_over20": float(np.mean([r["selected_goal_hop_count_over20m"] for r in metrics_rows])),
        "binding_pair": {"corrected":len(corrected),"damaged":len(damaged),"net":len(corrected)-len(damaged),
                         "corrected_with_public_geometry_change":sum(metrics_rows[i]["landmark_query_geometry_changed"] for i in corrected),
                         "damaged_with_public_geometry_change":sum(metrics_rows[i]["landmark_query_geometry_changed"] for i in damaged)},
        "target_categories": dict(Counter(r["target_category_proxy"] for r in metrics_rows)),
        "relation_grouping_is_heuristic": True,
        "relation_proxy": {g:{"n":sum(r["relation_complexity_proxy"]==g for r in metrics_rows),
                               "sr":float(np.mean([r["sr"] for r in metrics_rows if r["relation_complexity_proxy"]==g])) if any(r["relation_complexity_proxy"]==g for r in metrics_rows) else None,
                               "nms_top5":float(np.mean([r["initial_nms_recall_at_5"] for r in metrics_rows if r["relation_complexity_proxy"]==g])) if any(r["relation_complexity_proxy"]==g for r in metrics_rows) else None}
                          for g in sorted(set(r["relation_complexity_proxy"] for r in metrics_rows))},
        "posthoc_rgbd_proxy": {
            "episodes_with_target_contour": sum(x["target_contour_available"] for x in obs_summaries),
            "geometric_footprint_support_any_actual_frame": sum(x["first_geometric_projection_step"] is not None for x in obs_summaries),
            "max_projected_side_ge16px_any_actual_frame": sum(x["ever_long_side_16px"] for x in obs_summaries),
            "long32_short16px_proxy_any_actual_frame": sum(x["ever_long32_short16_proxy"] for x in obs_summaries),
            "depth_valid_target_pixels_any_actual_frame": sum(x["any_depth_valid_target_pixels"] for x in obs_summaries),
            "strictly_instance_recognizable": None,
            "warning":"Polygon footprint/orthophoto projection proxy; not real instance visibility or human/ML recognizability; no occlusion model.",
        },
        "closedloop_pair": run_metrics,
        "artifact_note":"V2 run used Geometry+Pose only; actual history RGB-D was not fed to policy. Field/Belief is static per query; only candidate trajectory selection can vary with pose.",
    }
    atomic_json(summary,out/"diagnostic_summary.json")
    write_dashboard(out,summary,metrics_rows,delta_rows)
    make_plots(out,summary,metrics_rows,obs_summaries)
    print(json.dumps({k:summary[k] for k in ("never_entered_20m","never_entered_topk_counts","never_entered_top5_contains_gt_but_policy_not_select","never_entered_top5_contains_gt_and_selected_but_route_missed","primary_failure_counts_failed_only","posthoc_rgbd_proxy","binding_pair")},ensure_ascii=False,indent=2),flush=True)


def make_plots(out, summary, rows, obs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax=plt.subplots(figsize=(10,5))
    failed=summary["primary_failure_counts_failed_only"]
    keys=list(failed); vals=[failed[k] for k in keys]
    ax.barh(keys[::-1],vals[::-1],color="#4c78a8")
    ax.set_xlabel("任务数（仅失败任务；主类别互斥）"); ax.grid(axis="x",alpha=.2)
    fig.tight_layout(); fig.savefig(out/"failure_primary_counts.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5))
    k=[1,5,10,20]
    allv=[100*summary["gt_nms_recall_all"][str(x)] for x in k]
    failv=[100*summary["never_entered_topk_recall"][str(x)] for x in k]
    ax.plot(k,allv,marker="o",label="全部 Val-Seen")
    ax.plot(k,failv,marker="o",label="从未进入 20m 的失败")
    ax.set_xticks(k); ax.set_ylabel("空间去重 Recall（%）"); ax.set_xlabel("Top-K（20m 空间 NMS）")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(out/"topk_recall.png",dpi=160); plt.close(fig)
    if obs:
        proxy=[x["max_long_side_px"] for x in obs if x["target_contour_available"]]
        fig,ax=plt.subplots(figsize=(7,4)); ax.hist(proxy,bins=30,color="#f58518")
        ax.set_xlabel("实际已访问视点中目标 Polygon 投影最长边（px）")
        ax.set_ylabel("Episode 数"); ax.set_title("仅为正射几何/像素投影代理，不代表实例可辨认")
        fig.tight_layout(); fig.savefig(out/"target_pixel_proxy.png",dpi=160); plt.close(fig)


def write_dashboard(out, summary, rows, delta_rows):
    from html import escape
    items=[]
    for x in delta_rows:
        idx=x["index"]
        items.append(f"<article class='card'><h3>任务 {idx} · {escape(str(x['map_name']))} · {escape(str(x['target_category_proxy']))}</h3>"
          f"<p><b>原指令：</b>{escape(str(x['instruction']))}</p><p>旧绑定：{escape(', '.join(x['legacy_anchor_names']) or '无')}；扩展绑定：{escape(', '.join(x['expanded_anchor_names']) or '无')}</p>"
          f"<p>旧SR={x['legacy_sr']} → 新SR={x['expanded_sr']}；NMS位置候选rank={x['initial_nms_gt_rank20m']}；终点NE={float(x['final_ne_m']):.1f}m</p>"
          f"<img loading='lazy' src='{escape(str(x['case_image']))}'><p>GT轮廓仅作事后几何代理，不是真实实例分割标签。</p></article>")
    failed = summary["primary_failure_counts_failed_only"]
    cards="".join(items)
    delta_index={int(x["index"]):x for x in delta_rows}
    table_rows=[]
    for x in rows:
        idx=int(x["index"]); rank=x.get("initial_nms_gt_rank20m")
        rank_text=">20 / 无" if rank is None else str(rank)
        case=delta_index.get(idx)
        case_link=f"<a href='{escape(str(case['case_image']))}'>查看配对图</a>" if case else "—"
        table_rows.append(
          f"<tr class='episode-row' data-text='{escape(str(x.get('instruction','')).lower())}'>"
          f"<td>{idx}</td><td>{escape(str(x.get('map_name','')))}</td><td>{escape(str(x.get('target_category_proxy','')))}</td>"
          f"<td>{int(bool(x.get('sr')))}</td><td>{int(bool(x.get('osr')))}</td><td>{rank_text}</td>"
          f"<td>{float(x.get('final_ne_m',0)):.1f}</td><td>{escape(str(x.get('primary_failure_class','')))}</td>"
          f"<td>{case_link}</td><td>{escape(str(x.get('instruction','')))}</td></tr>")
    table="".join(table_rows)
    html=f"""<!doctype html><meta charset='utf-8'><title>CityNav 失败归因</title>
<style>body{{font:15px system-ui;margin:24px;background:#f4f6f8;color:#17212b}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}}.card{{background:white;padding:16px;border-radius:12px;box-shadow:0 1px 6px #ccd}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:6px 10px;border:1px solid #ccd;vertical-align:top}}th{{position:sticky;top:0;background:#e9eef2}}td:last-child{{min-width:420px}}.scroll{{max-height:70vh;overflow:auto}}.summary{{background:#fff;padding:12px;margin:12px 0;border-radius:8px}}</style>
<h1>CityNav 2470 条 Val-Seen 闭环失败归因</h1><p>本页只展示扩展绑定相对旧绑定纠正/损坏的 {len(delta_rows)} 条任务。GT、目标Polygon、目标概率和像素投影均为事后诊断；不得视为模型推理输入或人工实例标注。</p>
<div class='summary'><b>完整覆盖：</b>SR={summary['closedloop_pair']['expanded']['SR']:.2%}（{summary['n']-sum(failed.values())}/{summary['n']}）；失败主类别互斥统计见图。该运行没有RGB输入，Geometry Belief静态。</div>
<img style='max-width:900px' src='failure_primary_counts.png'><img style='max-width:700px' src='topk_recall.png'>
<p><a href='per_episode_attribution.csv'>下载2470条逐任务归因CSV</a> · <a href='per_step_replay.csv'>下载逐决策回放CSV</a> · <a href='public_binding_corrected_damaged_74.csv'>下载绑定纠正/破坏案例CSV</a></p>
<label>搜索指令/场景/失败类别 <input id='search' oninput='filterRows()'></label><div class='scroll'><table><thead><tr><th>idx</th><th>地图</th><th>类</th><th>SR</th><th>OSR</th><th>空间NMS排名</th><th>NE(m)</th><th>主因</th><th>案例</th><th>指令</th></tr></thead><tbody>{table}</tbody></table></div>
<h2>绑定纠正/破坏配对案例</h2><div class='grid' id='cards'>{cards}</div>
<script>function filterRows(){{const q=document.getElementById('search').value.toLowerCase();document.querySelectorAll('.episode-row').forEach(x=>x.style.display=(x.innerText+' '+x.dataset.text).toLowerCase().includes(q)?'':'none');document.querySelectorAll('.card').forEach(x=>x.style.display=x.innerText.toLowerCase().includes(q)?'':'none')}}</script>"""
    (out/"index.html").write_text(html,encoding="utf-8")


if __name__ == "__main__":
    main()
