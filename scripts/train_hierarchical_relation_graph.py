"""TRAIN-only paired pilot for explicit coarse-to-fine spatial relation graph.

Example:
python scripts/train_hierarchical_relation_graph.py \
  --train-jsonl data/unified_full_train_revised.jsonl \
  --objects /path/to/cityrefer/objects.json \
  --geometry-checkpoint GraphNav/runs/unified_full_relation_s42_20261105/core/last.pt \
  --output runs/hrgb_train_s42 --max-queries 2000

Loads the original frozen Geometry from UnifiedPGVB; trains ONLY the new graph
head. All samples are TRAIN; held-out TRAIN-dev grouping is by physical target
ID and does not access Val/Test. Metrics are OFFLINE localization, never SR.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from graphnav.candidate_geometry import collate_full
from graphnav.checkpointing import file_sha256
from graphnav.full_relation_training import full_relation_inputs, load_full_relation_model
from graphnav.hierarchical_relation_graph import (
    HierarchicalRelationGraph, RegionAugmentedCandidateDataset,
)
from graphnav.static_localization import cross_entropy_goal
from graphnav.token_language import FrozenBERTTokens


def physical_group(record):
    # Used for TRAIN split isolation only; NEVER included in model features.
    parts = record.get("episode_id", "").split(":")
    return ":".join(parts[:2]) if len(parts) >= 2 else (
        record["map_name"] + ":" + record["instruction"])


def partition(records, seed, count, fraction=0.8):
    if not 0.0 < fraction < 1.0:
        raise ValueError("fit fraction must lie in (0,1)")
    chosen = list(range(len(records)))[:count] if count else list(range(len(records)))
    # Choose whole physical-object groups, preserving any multiple descriptions.
    groups = {}
    for i in chosen:
        groups.setdefault(physical_group(records[i]), []).append(i)
    fit, dev = [], []
    for name, indices in sorted(groups.items()):
        h = int(hashlib.sha256((str(seed) + "/" + name).encode()).hexdigest()[:12], 16)
        (fit if h % 1000 < int(fraction * 1000) else dev).extend(indices)
    if not fit or not dev:
        raise ValueError("empty TRAIN fit/dev partition; expand max-queries")
    if {physical_group(records[i]) for i in fit} & {
            physical_group(records[i]) for i in dev}:
        raise ValueError("physical target leakage across partitions")
    return fit, dev


def geometry_and_relation(geometry, graph, batch, bert, device, ablation):
    # Only whitelisted public features are passed through the forward path.
    inputs = full_relation_inputs(batch, bert, device)
    geometry.reset_episode()
    with torch.no_grad():
        base = geometry._geometry_field(
            inputs["static_inputs"], inputs["tokens"],
            inputs["token_mask"], inputs["node_emb"])
    result = graph(
        base_logits=base.belief_logits.detach().float(),
        candidate_context=base.dense_context.detach().float(),
        tokens=inputs["tokens"].detach().float(),
        token_mask=inputs["token_mask"],
        anchor_embeddings=inputs["node_emb"].detach().float(),
        anchor_valid=inputs["static_inputs"]["entity_valid"].bool(),
        pair_geometry=inputs["static_inputs"]["pair_geometry"].float(),
        region_grid_geometry=batch["region_grid_geometry"].to(device).float(),
        region_anchor_geometry=batch["region_anchor_geometry"].to(device).float(),
        region_valid=batch["region_valid"].to(device).bool(),
        region_types=batch["region_types"].to(device).long(),
        relation_mode=batch["relation_mode"].to(device).long(),
        ablation=ablation,
    )
    return base.belief_logits.detach().float(), result


@torch.no_grad()
def evaluate(geometry, graph, dataset, indices, bert, args):
    device = torch.device(args.device)
    graph.eval()
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size,
                        shuffle=False, collate_fn=collate_full, num_workers=0)
    samples = []
    for batch in loader:
        old, out = geometry_and_relation(
            geometry, graph, batch, bert, device, args.ablation)
        gt = batch["target_xy"].to(device).float()
        sizes = batch["map_sizes"].to(device).float()
        grid = batch["grid_xy"].to(device).float()
        pred = torch.stack((old.argmax(1), out["logits"].argmax(1)), 1)
        xy = grid.gather(1, pred[..., None].expand(-1, -1, 2))
        dist = ((xy - gt[:, None]) * sizes[:, None]).norm(dim=-1)
        mass = out["probabilities"].float()
        gt_dists = ((grid - gt[:, None]) * sizes[:, None]).norm(dim=-1)
        within20 = gt_dists <= 20.0
        for k, episode in enumerate(batch["episode_id"]):
            samples.append({
                "episode_id": episode, "geometry_NE_m": float(dist[k, 0]),
                "hierarchy_NE_m": float(dist[k, 1]),
                "geometry_R20": bool(dist[k, 0] <= 20.0),
                "hierarchy_R20": bool(dist[k, 1] <= 20.0),
                "delta_max": float(out["delta"][k].abs().max()),
                "delta_mean_abs": float(out["delta"][k].abs().mean()),
                "geometry_GT_mass20": float(
                    (old[k].softmax(-1) * within20[k]).sum()),
                "hierarchy_GT_mass20": float(
                    (mass[k] * within20[k]).sum()),
                "direct_active": bool(out["direct_active"][k]),
                "indirect_active": bool(out["indirect_active"][k]),
                "unresolved": bool(out["unresolved"][k]),
            })
    if not samples:
        raise ValueError("empty dev metrics")
    n = len(samples)
    old_ok = np.asarray([x["geometry_R20"] for x in samples], dtype=bool)
    new_ok = np.asarray([x["hierarchy_R20"] for x in samples], dtype=bool)
    metrics = {
        "n": n, "Geometry_R20": float(old_ok.mean()),
        "Hierarchy_R20": float(new_ok.mean()),
        "Geometry_NE_m": float(np.mean([x["geometry_NE_m"] for x in samples])),
        "Hierarchy_NE_m": float(np.mean([x["hierarchy_NE_m"] for x in samples])),
        "fixes": int((~old_ok & new_ok).sum()),
        "breaks": int((old_ok & ~new_ok).sum()),
        "mean_abs_belief_delta": float(np.mean(
            [x["delta_mean_abs"] for x in samples])),
        "direct_active": sum(x["direct_active"] for x in samples),
        "indirect_active": sum(x["indirect_active"] for x in samples),
        "unresolved": sum(x["unresolved"] for x in samples),
        "mean_GT20mass_change": float(np.mean([
            x["hierarchy_GT_mass20"] - x["geometry_GT_mass20"] for x in samples
        ])),
    }
    for name in ("direct_active", "indirect_active", "unresolved"):
        chosen = [x for x in samples if x[name]]
        metrics[name + "_R20"] = {
            "n": len(chosen),
            "geometry": float(np.mean([x["geometry_R20"] for x in chosen]))
                        if chosen else None,
            "hierarchy": float(np.mean([x["hierarchy_R20"] for x in chosen]))
                         if chosen else None,
        }
    return metrics, samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("train-jsonl", "objects", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--geometry-checkpoint", "--original-v2",
                        dest="geometry_checkpoint", type=Path, required=True,
                        help="Frozen original V2 or trained full-relation V1 bundle")
    parser.add_argument("--max-queries", type=int, default=2000,
                        help="TRAIN-only subset; 0 means all records")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-regions", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--fit-fraction", type=float, default=0.8)
    parser.add_argument("--ablation", choices=("full", "flat", "no-geometry", "no-language"),
                        default="full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.max_queries < 0 or args.epochs < 1 or args.batch_size < 1
            or args.lr <= 0):
        parser.error("invalid count, epoch, batch or learning rate")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no fake GPU benchmark")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    original = torch.load(args.geometry_checkpoint, map_location="cpu", weights_only=True)
    if original.get("format") not in (
        "unified-pgvb-v2", "unified-pgvb-full-relation-v1"
    ):
        raise ValueError("Expected original-v2 or trained full-relation-v1 checkpoint")
    geoargs = original["config"]["geometry_args"]
    dataset = RegionAugmentedCandidateDataset(
        args.train_jsonl, objects_json=args.objects, max_regions=args.max_regions,
        grid_size=28, max_landmarks=int(geoargs["max_landmarks"]),
        max_roads=int(geoargs["max_roads"]),
        max_relations=int(geoargs["max_relations"]),
        prior_sigma_m=float(geoargs["prior_sigma_m"]),
        rule_sigma_m=float(geoargs["rule_sigma_m"]),
        deduplicate_queries=False,
    )
    if {r.get("dataset_version") for r in dataset.records} != {"revised"}:
        raise ValueError("Only Revised TRAIN records are permitted")
    if any("teacher_future" in r or "uav_pose" in r for r in dataset.records):
        raise ValueError("Use static-only TRAIN; trajectories are not model input")
    fit, dev = partition(dataset.records, args.seed, args.max_queries,
                         args.fit_fraction)
    geometry = load_full_relation_model(original, device).eval().requires_grad_(False)
    bert = FrozenBERTTokens(device=args.device)
    graph = HierarchicalRelationGraph().to(device)
    optimizer = torch.optim.AdamW(graph.parameters(), lr=args.lr)
    args.output.mkdir(parents=True, exist_ok=True)
    expected = {
        "source_sha256": {name: file_sha256(getattr(args, name))
                          for name in ("train_jsonl", "objects", "geometry_checkpoint")},
        "fit_count": len(fit), "dev_count": len(dev),
        "seed": args.seed, "ablation": args.ablation,
        "epochs": args.epochs, "max_regions": args.max_regions,
        "isolation": "physical target grouped TRAIN fit/dev",
        "target_xy_forward_access": False, "val_test_access": False,
        "benchmark": "offline goal localization; NOT navigation SR",
    }
    started = time.perf_counter()
    losses = []
    for epoch in range(args.epochs):
        graph.train()
        loader = DataLoader(
            Subset(dataset, fit), batch_size=args.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(args.seed + epoch),
            collate_fn=collate_full, num_workers=0,
        )
        batch_losses = []
        for batch in loader:
            old, out = geometry_and_relation(
                geometry, graph, batch, bert, device, args.ablation)
            loss = cross_entropy_goal(
                out["logits"], batch["target_xy"].to(device))
            optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite TRAIN loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(graph.parameters(), 5.0)
            optimizer.step()
            batch_losses.append(float(loss.detach()))
        epoch_row = {"epoch": epoch + 1,
                     "loss": float(np.mean(batch_losses)),
                     "seconds": time.perf_counter() - started}
        losses.append(epoch_row)
        print(json.dumps(epoch_row), flush=True)

    metrics, cases = evaluate(geometry, graph, dataset, dev, bert, args)
    metadata = dict(expected, metrics=metrics, loss_history=losses,
                    evaluation_done_on="TRAIN physical-object-heldout dev only",
                    baseline="same frozen source Geometry checkpoint",
                    validation_not_executed_by_code_author=True)
    torch.save({
        "format": "citynav-hierarchical-explicit-relation-v1",
        "model": graph.state_dict(), "meta": metadata,
    }, args.output / "last.pt")
    (args.output / "summary.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output / "dev_cases.json").write_text(
        json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
