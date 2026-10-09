"""Stage 1 only: compare static Prior, Prior+Rules, Prior+Rules+TypedGAT.

Initial protocol intentionally has NO RGB, UAV pose, teacher trajectory or Stop.
Train only the GAT variant using target XY as a label; evaluate all methods
on EXACTLY the same data and denominators. val_unseen is never used for model
selection. Single NVIDIA RTX Pro 6000 96 GB: batch-size 16 by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from .foundations import FrozenBERT
from .static_localization import (
    StaticGoalLocalizer, StaticLocalizationDataset,
    collate_static, cross_entropy_goal, static_recall,
)


@dataclass
class StageSettings:
    grid_size: int = 28
    max_landmarks: int = 16
    max_roads: int = 12
    max_relations: int = 8
    hidden: int = 128
    text_dim: int = 768
    heads: int = 4
    prior_sigma_m: float = 25.0
    rule_sigma_m: float = 25.0
    rule_weight: float = 1.0


def load_data(path, settings, roads_json, batch_size, *, shuffle=False, workers=0):
    dataset = StaticLocalizationDataset(
        path, grid_size=settings.grid_size,
        max_landmarks=settings.max_landmarks, max_roads=settings.max_roads,
        max_relations=settings.max_relations, roads_json=roads_json,
        prior_sigma_m=settings.prior_sigma_m, rule_sigma_m=settings.rule_sigma_m,
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=workers, collate_fn=collate_static, pin_memory=True,
    )


def forward_batch(model, enc, batch, device, variant, use_amp):
    tensors = {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items() if isinstance(value, torch.Tensor)
    }
    if variant == "gat":
        instr = enc.encode(batch["text"]).to(device)
        b = len(batch["text"])
        max_nodes = len(batch["node_names"][0])
        all_names = [n for names in batch["node_names"] for n in names]
        node_emb = enc.encode(all_names).reshape(b, max_nodes, -1).to(device)
    else:
        instr = None
        node_emb = None
    # Never hand a target or any future-label tensor to model.forward().
    model_keys = (
        "nodes", "node_types", "node_valid", "edges", "adjacency",
        "prior", "rules", "relation_fields", "grid_xy",
    )
    model_inputs = {key: tensors[key] for key in model_keys}
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        logits = model(
            model_inputs, variant=variant, instruction_emb=instr, node_emb=node_emb
        )
    return logits, tensors


@torch.inference_mode()
def evaluate(model, enc, loader, device, use_amp, variants=("prior", "rules", "gat")):
    model.eval()
    stats = {
        v: dict(count=0, NE_sum_m=0., **{f"R@{k}/20m_count": 0 for k in (1, 4, 8)})
        for v in variants
    }
    source = dict(samples=0, with_relation=0, with_edges=0, with_roads=0, with_road_relation=0)
    for batch in loader:
        source["samples"] += len(batch["text"])
        source["with_relation"] += int((batch["relation_count"] > 0).sum())
        source["with_edges"] += int((batch["edge_count"] > 0).sum())
        source["with_roads"] += int((batch["road_count"] > 0).sum())
        source["with_road_relation"] += int((batch["road_relation_count"] > 0).sum())
        for variant in variants:
            logits, tensors = forward_batch(model, enc, batch, device, variant, use_amp)
            result = static_recall(
                logits, tensors["target_xy"], tensors["map_sizes"],
                tensors["grid_xy"]
            )
            for key, value in result.items():
                stats[variant][key] += value
    summary = {}
    for name, values in stats.items():
        n = max(1, values["count"])
        summary[name] = dict(
            samples=values["count"],
            NE_m=values["NE_sum_m"] / n,
            **{f"R@{k}/20m": values[f"R@{k}/20m_count"] / n for k in (1, 4, 8)}
        )
    return {
        "variants": summary,
        "coverage": {
            "samples": source["samples"],
            "grounded_relation_fraction": source["with_relation"] / max(1, source["samples"]),
            "roads_available_fraction": source["with_roads"] / max(1, source["samples"]),
            "grounded_road_relation_fraction": source["with_road_relation"] / max(1, source["samples"]),
        },
    }


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    settings = StageSettings(
        grid_size=args.grid_size, max_landmarks=args.max_landmarks,
        max_roads=args.max_roads, max_relations=args.max_relations,
        hidden=args.hidden, heads=args.heads,
        prior_sigma_m=args.prior_sigma_m, rule_sigma_m=args.rule_sigma_m,
        rule_weight=args.rule_weight,
    )
    device = torch.device(args.device)
    amp = device.type == "cuda"
    loaders = {
        "train_seen": load_data(args.train, settings, args.roads_json, args.batch_size,
                                shuffle=True, workers=args.workers),
        "val_seen": load_data(args.val_seen, settings, args.roads_json, args.batch_size,
                             workers=args.workers),
        "val_unseen": load_data(args.val_unseen, settings, args.roads_json, args.batch_size,
                               workers=args.workers),
    }
    encoder = FrozenBERT(name=args.text_backbone, device=args.device)
    model = StaticGoalLocalizer(
        hidden=settings.hidden, text_dim=settings.text_dim, heads=settings.heads,
        relations=settings.max_relations, rule_weight=settings.rule_weight,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    best_seen = -1.
    for epoch in range(args.epochs):
        model.train()
        losses = []
        began = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for step, batch in enumerate(loaders["train_seen"]):
            opt.zero_grad(set_to_none=True)
            logits, tensor = forward_batch(model, encoder, batch, device, "gat", amp)
            loss = cross_entropy_goal(
                logits, tensor["target_xy"], settings.grid_size
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            if args.max_train_batches and step + 1 >= args.max_train_batches:
                break
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - began
        peak_gib = (torch.cuda.max_memory_allocated(device) / (1024**3)
                    if device.type == "cuda" else None)
        seen = evaluate(model, encoder, loaders["val_seen"], device, amp)
        performance = seen["variants"]["gat"]["R@8/20m"]
        record = {
            "epoch": epoch + 1,
            "train_goal_loss": float(np.mean(losses)),
            "train_seconds": train_seconds,
            "samples_per_second": min(
                len(loaders["train_seen"].dataset), len(losses) * args.batch_size
            ) / max(train_seconds, 1e-6),
            "gpu_peak_gib": peak_gib,
            "val_seen": seen,
        }
        records.append(record)
        if performance > best_seen:
            best_seen = performance
            torch.save({
                "model": model.state_dict(), "settings": asdict(settings),
                "selected_on": "val_seen_R@8/20m", "epoch": epoch+1,
                "text_backbone": args.text_backbone,
            }, output / "best.pt")
        print(json.dumps({
            "epoch": epoch + 1, "train_goal_loss": record["train_goal_loss"],
            "val_seen": seen,
        }, ensure_ascii=False), flush=True)
    # Select ONLY on val_seen. Unseen is an independently held-out report.
    ckpt = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    final = {
        "settings": asdict(settings),
        "hardware": args.hardware,
        "batch_size": args.batch_size,
        "selected_epoch": ckpt["epoch"],
        "val_seen": evaluate(model, encoder, loaders["val_seen"], device, amp),
        "val_unseen": evaluate(model, encoder, loaders["val_unseen"], device, amp),
        "history": records,
        "note": "Offline static goal localization only. No RGB, UAV control, SR or SPL.",
    }
    (output / "static_localization_metrics.json").write_text(
        json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output / "static_localization_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "split", "variant", "samples", "R@1/20m", "R@4/20m", "R@8/20m", "NE_m"
        ])
        writer.writeheader()
        for split in ("val_seen", "val_unseen"):
            for variant, values in final[split]["variants"].items():
                writer.writerow(dict(split=split, variant=variant, **values))
    print("Saved " + str(output / "static_localization_metrics.csv"))
    return final


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", required=True)
    p.add_argument("--val-seen", required=True)
    p.add_argument("--val-unseen", required=True)
    p.add_argument("--roads-json")
    p.add_argument("--output", default="runs/static_graph")
    p.add_argument("--hardware", default="RTX Pro 6000 96GB")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--text-backbone", default="bert-base-uncased")
    p.add_argument("--grid-size", type=int, default=28)
    p.add_argument("--max-landmarks", type=int, default=16)
    p.add_argument("--max-roads", type=int, default=12)
    p.add_argument("--max-relations", type=int, default=8)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--prior-sigma-m", type=float, default=25.)
    p.add_argument("--rule-sigma-m", type=float, default=25.)
    p.add_argument("--rule-weight", type=float, default=1.)
    train(p.parse_args())


if __name__ == "__main__":
    main()
