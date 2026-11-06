"""Prepare and locally fit UnifiedPGVB-V3 on TRAIN; NO navigation evaluation.

Cache uses only public named-landmark geometry to choose deterministic
fixed-yaw RGB-D vantage points. Full original instructions, *one* frozen
SigLIP2 ViT (image patches + text) and frozen BERT tokens are cached.
Target GT Polygon and GT orthophoto crop embedding go into a physically
separate PRIVATE label cache, never into the student or viewpoint planner.

Examples and provenance requirements: docs/UNIFIED_PGVB_V3.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from graphnav.appearance_matching import ObservedSigLIPMatchingEncoder
from graphnav.candidate_geometry import FullCandidateDataset, collate_full
from graphnav.checkpointing import atomic_json, atomic_save, file_sha256
from graphnav.full_experiment import _batch_inputs, new_model
from graphnav.memory import ObservedRGBMemory
from graphnav.renderer import CityNavObservedViewRenderer
from graphnav.token_language import FrozenBERTTokens
from graphnav.unified_pgvb import UnifiedPGVB
from graphnav.unified_pgvb_v3 import (
    UnifiedPGVBV3, v3_losses, offline_topk, SIZE, PATCHES,
)


STATIC_KEYS = (
    "nodes", "node_types", "node_valid", "entity_valid", "adjacency",
    "edges", "prior", "prior_buildings", "rules", "relation_fields",
    "pair_geometry", "grid_xy",
)
BUNDLE_FORMAT = "unified-pgvb-v2"
V3_FORMAT = "unified-pgvb-v3-decoupled-rgbd-v1"


def _load_base(path, device):
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if bundle.get("format") != BUNDLE_FORMAT:
        raise ValueError("V3 first pilot requires verified original UnifiedPGVB-V2")
    args = SimpleNamespace(**bundle["config"]["geometry_args"])
    if int(args.grid_size) != 28 or int(args.hidden) != 128:
        raise ValueError("expected original 28x28 Geometry with 128-d context")
    geometry = new_model(args, torch.device(device))
    del geometry.vision_proj
    del geometry.visual_score
    model = UnifiedPGVB(geometry=geometry)
    model.load_state_dict(bundle["model"], strict=True)
    return model.geometry, args


def fixed_yaw_public_poses(record, *, ground, altitude_m, yaw, offset_m, views):
    """Only public named reference Polygon & public bounds; never target_xy.

    Pose lies at named landmark center and four offsets, fixed world yaw.
    Map ROI may be wider than published Polygon due to raster-view footprint.
    """
    if views < 1 or views > 5 or altitude_m <= 0 or offset_m <= 0:
        raise ValueError("fixed-yaw pilot requires 1..5 views and positive scales")
    if not math.isfinite(yaw):
        raise ValueError("invalid fixed yaw")
    bounds = np.asarray(record["world_bounds"], np.float64)
    if bounds.shape != (4,) or (bounds[2:] <= bounds[:2]).any():
        raise ValueError("invalid public map bounds")
    lm = record.get("landmarks", [])
    if not lm:
        return []   # Unknown, not GT-selected fallback.
    p = np.asarray(lm[0].get("contour", []), np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.isfinite(p).all():
        return []
    # CityNav TRAIN JSONL public landmark polygons use normalized XY.
    if (p < -1e-4).any() or (p > 1+1e-4).any():
        raise ValueError("expected public normalized landmark contour")
    center = bounds[:2] + p.mean(0) * (bounds[2:] - bounds[:2])
    offsets = [(0, 0), (offset_m, 0), (0, offset_m),
               (-offset_m, 0), (0, -offset_m)][:views]
    poses = []
    for dx, dy in offsets:
        xy = np.clip(center + (dx, dy), bounds[:2] + .1, bounds[2:] - .1)
        xyz = np.asarray((xy[0], xy[1], float(ground)+altitude_m, yaw),
                         dtype=np.float64)
        if not any(np.linalg.norm(xyz[:2]-q[:2]) < 1e-3 for q in poses):
            poses.append(xyz)
    if any(abs(p[3]-poses[0][3]) > 1e-8 for p in poses):
        raise AssertionError("yaw must remain constant")
    return poses


def _depth_patch_tensor(obs):
    valid = np.asarray(obs["depth_valid"], dtype=bool)
    depth = np.asarray(obs["depth_m"], dtype=np.float32)
    if valid.shape != (384, 384) or depth.shape != valid.shape:
        raise ValueError("expected native 384 RGB-D")
    vis = valid.reshape(PATCHES, 16, PATCHES, 16).mean((1, 3))
    d = np.where(valid, depth, 0).reshape(PATCHES, 16, PATCHES, 16)
    avg = d.sum((1, 3)) / np.maximum(vis*256, 1)
    features = np.stack((np.clip(avg, 0, 300)/100, vis), -1).reshape(-1, 2)
    return torch.from_numpy(features).float(), torch.from_numpy(
        (vis >= .90).reshape(-1))


def _teacher_crop(renderer, record, polygon):
    """Private-only crop from TRAIN orthophoto. Never call in model.forward."""
    if polygon is None or len(polygon) < 3:
        return None
    renderer._open(record["map_name"])
    coords = []
    for x, y in polygon:
        row, col = renderer._raster.index(float(x), float(y))
        coords.append((col, row))
    pts = np.asarray(coords, np.int32)
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    if x1 <= x0 or y1 <= y0:
        return None
    margin = int(max(4, .2 * max(x1-x0, y1-y0)))
    h, w = renderer._rgb.shape[:2]
    x0, x1 = max(0, x0-margin), min(w, x1+margin)
    y0, y1 = max(0, y0-margin), min(h, y1+margin)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = renderer._rgb[y0:y1, x0:x1]
    if crop.size < 64:
        return None
    return cv2.resize(crop, (384, 384), interpolation=cv2.INTER_AREA)


def _cache_one(index, record, sample, encoder, native, renderer,
               public_objects, ground, args, out):
    # These IDs are used exclusively to look up TRAIN *labels*, AFTER legal
    # observation poses have been selected from record.landmarks.
    ids = str(record.get("episode_id", "")).split(":")
    if len(ids) < 4 or ids[0] != record["map_name"]:
        raise ValueError("TRAIN record must identify its physical target")
    obj_id = ids[1]
    obj = public_objects.get(record["map_name"], {}).get(obj_id)
    polygon = obj.get("contour") if obj else None
    from graphnav.semantic_geometry import valid_surface_polygon
    if polygon is not None and not valid_surface_polygon(np.asarray(polygon)):
        polygon = None

    batch = collate_full([sample])
    static, bert, bert_mask, nodes = _batch_inputs(
        batch, encoder, args.device, "geometry")
    poses = fixed_yaw_public_poses(
        record, ground=ground, altitude_m=args.altitude_m,
        yaw=args.fixed_yaw, offset_m=args.offset_m, views=args.views)
    if not poses:
        # One PADDED non-observation. Valid mask is zero, so vision is absent.
        bounds = np.asarray(record["world_bounds"], dtype=np.float64)
        xy = (bounds[:2]+bounds[2:]) / 2
        poses = [np.asarray((xy[0], xy[1], ground+args.altitude_m,
                             args.fixed_yaw), np.float64)]
        genuine = False
    else:
        genuine = True
    features, depth_features, masks, footprints, actual_poses = [], [], [], [], []
    for pose in poses:
        fp = ObservedRGBMemory.view_corners(
            pose[:2], pose[3], pose[2]-ground)
        if genuine:
            obs = renderer.current_rgbd(
                record["map_name"], pose, ground)
            dp, valid = _depth_patch_tensor(obs)
            rgb = native.processor(
                images=[cv2.cvtColor(obs["bgr"], cv2.COLOR_BGR2RGB)],
                return_tensors="pt").to(native.device)
            with torch.inference_mode(), torch.autocast(
                    "cuda", dtype=torch.bfloat16,
                    enabled=native.device.type == "cuda"):
                image = native.model.vision_model(**rgb).last_hidden_state[0]
            if image.shape != (PATCHES*PATCHES, native.dimension):
                raise ValueError("SigLIP2-Large/384 dense token grid mismatch")
            features.append(image.detach().cpu().half().clone())
        else:
            dp = torch.zeros(PATCHES*PATCHES, 2)
            valid = torch.zeros(PATCHES*PATCHES, dtype=torch.bool)
            features.append(torch.zeros(PATCHES*PATCHES, native.dimension,
                                        dtype=torch.float16))
        depth_features.append(dp.half())
        masks.append(valid)
        footprints.append(torch.tensor(fp, dtype=torch.float32))
        actual_poses.append(torch.tensor(pose, dtype=torch.float32))
    # Pad to fixed F; padding must retain nondegenerate footprint and zero mask.
    while len(features) < args.views:
        features.append(torch.zeros_like(features[0]))
        depth_features.append(torch.zeros_like(depth_features[0]))
        masks.append(torch.zeros_like(masks[0]))
        footprints.append(footprints[0].clone())
        actual_poses.append(actual_poses[0].clone())
    with torch.inference_mode():
        text_embedding = native.texts([record["instruction"]])[0].cpu().half().clone()
        crop = _teacher_crop(renderer, record, polygon)
        teacher = (native.images([crop])[0].cpu().half().clone()
                   if crop is not None
                   else torch.zeros(native.dimension, dtype=torch.float16))

    public = dict(
        episode_id=record["episode_id"],
        map_name=record["map_name"],
        static={key:static[key][0].detach().cpu().clone() for key in STATIC_KEYS},
        bert_tokens=bert[0].detach().cpu().half().clone(),
        bert_mask=bert_mask[0].detach().cpu().clone(),
        landmark_embeddings=nodes[0].detach().cpu().half().clone(),
        siglip_text=text_embedding, rgb_patches=torch.stack(features),
        depth_patches=torch.stack(depth_features),
        patch_valid=torch.stack(masks), footprints_world=torch.stack(footprints),
        frame_poses=torch.stack(actual_poses),
        world_bounds=torch.tensor(record["world_bounds"], dtype=torch.float32),
        observation_available=bool(genuine and any(m.any() for m in masks)),
    )
    private = dict(
        target_xy=sample["target_xy"].cpu().float(),
        target_polygon=polygon,
        teacher_image=teacher,
    )
    key = f"{index:06d}"
    atomic_save(public, out/"public"/f"{key}.pt")
    atomic_save(private, out/"private"/f"{key}.pt")
    return dict(index=index, episode_id=record["episode_id"],
                observations=sum(int(m.any()) for m in masks),
                polygon_valid=polygon is not None,
                teacher_available=bool(teacher.abs().sum() > 0))


def cache(args):
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    if args.views < 1 or args.views > 5:
        raise ValueError("views must be between 1 and 5")
    root = args.cache
    (root/"public").mkdir(parents=True, exist_ok=True)
    (root/"private").mkdir(parents=True, exist_ok=True)
    base = torch.load(args.source_v2, map_location="cpu", weights_only=True)
    if base.get("format") != BUNDLE_FORMAT:
        raise ValueError("source_v2 must be the VERIFIED V2 checkpoint")
    cfg = SimpleNamespace(**base["config"]["geometry_args"])
    dataset = FullCandidateDataset(
        args.train_jsonl, grid_size=28, max_landmarks=cfg.max_landmarks,
        max_roads=cfg.max_roads, max_relations=cfg.max_relations,
        prior_sigma_m=cfg.prior_sigma_m, rule_sigma_m=cfg.rule_sigma_m,
        deduplicate_queries=False)
    if any(r.get("dataset_version") != "revised" for r in dataset.records):
        raise ValueError("V3 pilot accepts Revised TRAIN only")
    objects = json.loads(args.objects.read_text(encoding="utf-8"))
    ground = json.loads(args.ground_levels.read_text(encoding="utf-8"))
    manifest = {
        "format": "unified-pgvb-v3-cache-v1",
        "source": {key:file_sha256(getattr(args, key)) for key in (
            "train_jsonl", "objects", "ground_levels", "source_v2")},
        "rgbd_dir": str(args.rgbd_dir.resolve()),
        "siglip2": "google/siglip2-large-patch16-384",
        "cache_scope": "Revised TRAIN only", "seed":42,
        "views":args.views, "fixed_yaw":args.fixed_yaw,
        "altitude_m":args.altitude_m, "offset_m":args.offset_m,
        "selection": "first public named landmark, never target GT",
        "no_future_frames": True, "gt_disjoint_from_forward": True,
        "is_true_perspective_depth": False,
    }
    file = root/"manifest.json"
    if file.exists() and json.loads(file.read_text()) != manifest:
        raise ValueError("Cache source/protocol changed; use new directory")
    atomic_json(manifest, file)
    encoder = FrozenBERTTokens(cfg.text_backbone, device=args.device,
                               max_length=cfg.max_text_length)
    native = ObservedSigLIPMatchingEncoder(device=args.device, microbatch=4)
    renderer = CityNavObservedViewRenderer(
        args.rgbd_dir, root/"renderer_unused", image_size=384)
    limit = len(dataset) if args.limit == 0 else min(args.limit, len(dataset))
    stats = []
    for index in range(limit):
        path = root/"public"/f"{index:06d}.pt"
        lbl = root/"private"/f"{index:06d}.pt"
        if path.exists() and lbl.exists():
            continue
        r = dataset.records[index]
        if r["map_name"] not in ground:
            raise ValueError(f"No ground-level metadata for {r['map_name']}")
        stats.append(_cache_one(index, r, dataset[index], encoder, native,
                                renderer, objects, float(ground[r["map_name"]]),
                                args, root))
        if index % 20 == 0:
            print(json.dumps({"cached":index+1, "limit":limit}), flush=True)
    atomic_json({"cached_files":len(list((root/"public").glob("*.pt"))),
                 "requested":limit, "new":stats},root/"cache_summary.json")
    print(f"V3 TRAIN cache ready: {limit} requested; no Val/Test used")


class V3CacheDataset(Dataset):
    def __init__(self, directory, *, max_queries=0):
        self.root = Path(directory)
        manifest = json.loads((self.root/"manifest.json").read_text())
        if manifest.get("format") != "unified-pgvb-v3-cache-v1":
            raise ValueError("Not a V3 TRAIN cache")
        self.manifest = manifest
        paths = sorted((self.root/"public").glob("*.pt"))
        if max_queries:
            paths = paths[:max_queries]
        self.keys = []
        for p in paths:
            if not (self.root/"private"/p.name).is_file():
                raise FileNotFoundError(f"Missing private train label: {p.name}")
            self.keys.append(p.stem)
        if len(self.keys) < 2:
            raise ValueError("Need >=2 prepared TRAIN queries")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        key = self.keys[i]
        public = torch.load(self.root/"public"/f"{key}.pt", weights_only=True)
        private = torch.load(self.root/"private"/f"{key}.pt", weights_only=True)
        return public, private


def _collate(pairs, device):
    p = [x[0] for x in pairs]
    q = [x[1] for x in pairs]
    l = max(row["bert_tokens"].shape[0] for row in p)
    tokens = torch.zeros(len(p), l, 768)
    mask = torch.zeros(len(p), l, dtype=torch.bool)
    for i, row in enumerate(p):
        n = row["bert_tokens"].shape[0]
        tokens[i, :n] = row["bert_tokens"].float()
        mask[i, :n] = row["bert_mask"]
    kwargs = dict(
        static_inputs={k: torch.stack([v["static"][k] for v in p]).to(device)
                       for k in STATIC_KEYS},
        bert_tokens=tokens.to(device), bert_mask=mask.to(device),
        landmark_embeddings=torch.stack([v["landmark_embeddings"].float()
                                         for v in p]).to(device),
        siglip_text=torch.stack([v["siglip_text"].float() for v in p]).to(device),
        rgb_patches=torch.stack([v["rgb_patches"].float() for v in p]).to(device),
        depth_patches=torch.stack([v["depth_patches"].float() for v in p]).to(device),
        patch_valid=torch.stack([v["patch_valid"] for v in p]).to(device),
        footprints_world=torch.stack([v["footprints_world"] for v in p]).to(device),
        frame_poses=torch.stack([v["frame_poses"] for v in p]).to(device),
        world_bounds=torch.stack([v["world_bounds"] for v in p]).to(device),
    )
    private = dict(
        target_xy=torch.stack([v["target_xy"] for v in q]).to(device),
        target_polygons=[v["target_polygon"] for v in q],
        teacher_features=torch.stack(
            [v["teacher_image"].float() for v in q]).to(device),
    )
    return kwargs, private


def _physical_id(episode_id):
    parts = episode_id.split(":")
    return ":".join(parts[:2])


def fit_dev_partition(ds, seed=42, fraction=.8):
    """Physical-target split; no overlap of object ID between TRAIN fit/dev."""
    fit, dev = [], []
    for i in range(len(ds)):
        p, _ = ds[i]
        group = _physical_id(p["episode_id"])
        r = int(hashlib.sha256(f"{seed}/{group}".encode()).hexdigest()[:8], 16)
        (fit if r % 1000 < fraction*1000 else dev).append(i)
    if not fit or not dev:
        raise ValueError("TRAIN group split produced empty partition")
    return fit, dev


@torch.no_grad()
def evaluate(model, ds, indices, device, batch_size):
    """Geometry-only vs V3 fused, same held-out TRAIN physical IDs."""
    from statistics import mean
    model.eval()
    records = []
    for offset in range(0, len(indices), batch_size):
        pairs = [ds[i] for i in indices[offset:offset+batch_size]]
        kwargs, labels = _collate(pairs, device)
        result = model(**kwargs)
        geo = offline_topk(result, labels["target_xy"],
                           kwargs["world_bounds"], branch="geometry")
        fused = offline_topk(result, labels["target_xy"],
                             kwargs["world_bounds"], branch="fused")
        for i, (p, _) in enumerate(pairs):
            records.append(dict(
                episode_id=p["episode_id"],
                geometry_NE_m=geo["Top1_NE_m"][i],
                fused_NE_m=fused["Top1_NE_m"][i],
                geometry_R={str(k):bool(geo["hit_flags"][str(k)][i])
                            for k in (1,5,20)},
                fused_R={str(k):bool(fused["hit_flags"][str(k)][i])
                         for k in (1,5,20)},
                geometry_top1_xy=geo["Top1_xy"][i],
                fused_top1_xy=fused["Top1_xy"][i],
                observed_pixels=int(result.observed_mask[i].sum()),
                valid_rgb_patches=int(result.patch_valid[i].sum()),
            ))
    if not records:
        raise ValueError("No TRAIN heldout cases")
    metrics = dict(
        n=len(records),
        Geometry_NE_m=mean(x["geometry_NE_m"] for x in records),
        Fused_NE_m=mean(x["fused_NE_m"] for x in records),
        observed_queries=sum(x["observed_pixels"] > 0 for x in records),
        no_visual_queries=sum(x["observed_pixels"] == 0 for x in records),
    )
    for k in (1,5,20):
        old = [x["geometry_R"][str(k)] for x in records]
        new = [x["fused_R"][str(k)] for x in records]
        metrics[f"Geometry_R@{k}/20m"] = mean(old)
        metrics[f"Fused_R@{k}/20m"] = mean(new)
        metrics[f"R@{k}_fixes"] = sum(not a and b for a,b in zip(old,new))
        metrics[f"R@{k}_breaks"] = sum(a and not b for a,b in zip(old,new))
    return metrics, records

def train(args):
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use local GPU for this pilot")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    ds = V3CacheDataset(args.cache, max_queries=args.max_queries)
    if ds.manifest["source"]["source_v2"] != file_sha256(args.source_v2):
        raise ValueError("Prepared cache and model provenance disagree")
    fit, dev = fit_dev_partition(ds, args.seed)
    geom, cfg = _load_base(args.source_v2, args.device)
    model = UnifiedPGVBV3(geom).to(args.device)
    opt = torch.optim.AdamW([
        dict(params=list(model.geometry.parameters()), lr=args.lr_geometry),
        dict(params=[p for n,p in model.named_parameters()
                     if not n.startswith("geometry.")], lr=args.lr_visual),
    ], weight_decay=0.001)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/"last.pt").exists():
        raise FileExistsError("Do not silently overwrite V3 experimental weights")
    steps = []
    for epoch in range(args.epochs):
        model.train()
        order = np.random.default_rng(args.seed+epoch).permutation(fit).tolist()
        total = 0.0
        for at in range(0, len(order), args.batch_size):
            batch = [ds[i] for i in order[at:at+args.batch_size]]
            kwargs, labels = _collate(batch, args.device)
            output = model(**kwargs)
            loss = v3_losses(
                output, labels["target_xy"], labels["target_polygons"],
                kwargs["world_bounds"], labels["teacher_features"],
                model.teacher_head)
            if not torch.isfinite(loss["total"]):
                raise FloatingPointError("nonfinite V3 loss")
            opt.zero_grad(set_to_none=True)
            loss["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0,
                                           error_if_nonfinite=True)
            opt.step()
            total += float(loss["total"].detach()) * len(batch)
        report = dict(epoch=epoch+1,fit=len(fit),
                      mean_train_loss=total/len(fit))
        steps.append(report)
        print(json.dumps(report), flush=True)
    metrics, cases = evaluate(model, ds, dev, args.device, args.batch_size)
    manifest = dict(
        format=V3_FORMAT, source_v2_sha256=file_sha256(args.source_v2),
        cache_manifest_sha256=file_sha256(args.cache/"manifest.json"),
        train_split="Revised TRAIN physical-ID fit/dev", seed=args.seed,
        fit=len(fit), dev=len(dev), epochs=args.epochs, losses=steps,
        offline_metrics=metrics, visual_encoder_frozen=True,
        visual_backbone="google/siglip2-large-patch16-384",
        policy_controller_used=False, val_test_access=False,
        closed_loop_SR_not_measured=True,
    )
    atomic_save(dict(format=V3_FORMAT, model=model.state_dict(),
                     manifest=manifest), args.output/"last.pt")
    atomic_json(dict(manifest=manifest, paired=cases),
                args.output/"report.json")
    print(json.dumps(metrics, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    a = sub.add_parser("cache", help="Cache legally sampled TRAIN RGB-D and GT-only labels")
    for name in ("train-jsonl", "objects", "ground-levels", "rgbd-dir",
                 "source-v2", "cache"):
        a.add_argument("--"+name, type=Path, required=True)
    a.add_argument("--limit", type=int, default=128)
    a.add_argument("--views", type=int, default=5)
    a.add_argument("--fixed-yaw", type=float, default=0.0,
                   help="Fixed engineering-world yaw in radians; not geographic north")
    a.add_argument("--altitude-m", type=float, default=80.0)
    a.add_argument("--offset-m", type=float, default=35.0)
    a.add_argument("--device", default="cuda")
    b = sub.add_parser("train", help="TRAIN-only pilot with physical-target heldout dev")
    for name in ("cache", "source-v2", "output"):
        b.add_argument("--"+name, type=Path, required=True)
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--max-queries", type=int, default=0)
    b.add_argument("--epochs", type=int, default=2)
    b.add_argument("--batch-size", type=int, default=1)
    b.add_argument("--lr-geometry", type=float, default=1e-5)
    b.add_argument("--lr-visual", type=float, default=1e-4)
    b.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.action == "cache":
        cache(args)
    else:
        if args.epochs < 1 or args.batch_size < 1 or args.max_queries < 0:
            parser.error("invalid training arguments")
        train(args)


if __name__ == "__main__":
    main()
