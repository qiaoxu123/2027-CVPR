"""Controlled A0-A5 CityNav localization benchmark on single RTX Pro 6000 96GB.

A0 building_prior; A1 prior+rules; A2 complete BERT tokens;
A3 + each candidate-to-contour geometry; A4 + typed GATv2;
A5 + observed-only RGB/SigLIP2 reranking (GAT checkpoint frozen).

No online SR/SPL claim: this is goal-localization retrieval. Revised and
Original data CANNOT silently mix. Frozen RGB and text encoders are cached.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .candidate_geometry import FullCandidateDataset, collate_full
from .full_localization import FullTokenGeometryLocalizer
from .token_language import FrozenBERTTokens
from .static_localization import cross_entropy_goal, static_recall
from .foundations import FrozenSigLIP2, CandidateVisionCache
from .visual_supervision import visual_candidate_batch


TRAINED = ("tokens", "geometry", "gat", "rgb")
BASELINES = ("building_prior", "prior", "rules")
METRIC_KEYS = ("R@1/20m", "R@4/20m", "R@8/20m")


def _query_keys(dataset):
    return {(x.get("map_name", ""), x["instruction"]) for x in dataset.records}


def make_loader(path, args, *, shuffle=False, visual_required=False):
    dataset = FullCandidateDataset(
        path, grid_size=args.grid_size, max_landmarks=args.max_landmarks,
        max_roads=args.max_roads, max_relations=args.max_relations,
        roads_json=args.roads_json, prior_sigma_m=args.prior_sigma_m,
        rule_sigma_m=args.rule_sigma_m, visual_required=visual_required,
    )
    versions = {r.get("dataset_version", "unspecified") for r in dataset.records}
    matches = {r.get("match_mode", "unspecified") for r in dataset.records}
    if versions != {args.dataset_version} or matches != {args.match_mode}:
        raise ValueError(
            f"Cannot mix protocol. Requested {args.dataset_version}/{args.match_mode}; "
            f"file contains versions={versions}, match_modes={matches}"
        )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.workers, collate_fn=collate_full,
        pin_memory=args.device.startswith("cuda"),
    )
    return loader


def _batch_inputs(batch, encoder, device, variant):
    keys = (
        "nodes", "node_types", "node_valid", "entity_valid", "adjacency",
        "edges", "prior", "prior_buildings", "rules", "relation_fields",
        "pair_geometry", "grid_xy",
    )
    observable = {k: batch[k].to(device, non_blocking=True) for k in keys}
    if variant in BASELINES:
        return observable, None, None, None
    # Frozen BERT token sequence is never replaced by a CLS-only embedding.
    states, mask = encoder.encode_tokens(batch["text"])
    states = states.to(device).clone()
    mask = mask.to(device)
    node_names = [name for group in batch["node_names"] for name in group]
    node_emb = None
    if variant != "tokens":
        raw = encoder.encode(node_names)
        node_emb = raw.reshape(len(batch["text"]), len(batch["node_names"][0]), -1)
        node_emb = node_emb.to(device).clone()
    return observable, states, mask, node_emb


def run_model(model, batch, encoder, vision, image_cache, args, variant, device):
    observable, tokens, mask, nodes = _batch_inputs(batch, encoder, device, variant)
    with torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")
    ):
        field = model(
            observable, variant=variant,
            tokens=tokens, token_mask=mask, node_emb=nodes,
        )
        if variant == "rgb":
            embeddings, confidence = visual_candidate_batch(
                field, batch["rgb_observations"], batch["world_bounds"],
                vision, vision_dim=args.vision_dim,
                min_coverage=args.min_visual_coverage,
                crop_extent_m=args.crop_extent_m,
                cache=image_cache,
            )
            # Only trusted evidence from the observed causal mosaic enters logits.
            field = model.refine_rgb(
                field, embeddings.to(device), confidence.to(device),
                threshold=args.min_visual_coverage,
            )
    return field


@torch.inference_mode()
def evaluate(model, loader, encoder, vision, cache, args, variant, device):
    model.eval()
    total = {"samples": 0, "NE_sum_m": 0.,
             **{f"R@{k}/20m_count": 0 for k in (1, 4, 8)}}
    observed, proposals, samples_with_visual, bound, relation = 0, 0, 0, 0, 0
    for batch in loader:
        field = run_model(model, batch, encoder, vision, cache, args, variant, device)
        metrics = static_recall(
            field.belief_logits.float(),
            batch["target_xy"].to(device),
            batch["map_sizes"].to(device),
            batch["grid_xy"].to(device),
        )
        for key, value in metrics.items():
            total[key] += value
        bound += int((batch["surface_count"] > 0).sum())
        relation += int((batch["relation_count"] > 0).sum())
        if field.visual_coverage is not None:
            mask = field.visual_coverage >= args.min_visual_coverage
            observed += int(mask.sum())
            samples_with_visual += int(mask.any(dim=1).sum())
            proposals += int(mask.numel())
    n = max(1, total["samples"])
    result = {
        "samples": total["samples"],
        "NE_m": total["NE_sum_m"] / n,
        **{f"R@{k}/20m": total[f"R@{k}/20m_count"] / n for k in (1, 4, 8)},
        "surface_anchor_fraction": bound / n,
        "relation_grounded_fraction": relation / n,
    }
    if variant == "rgb":
        result.update(
            visual_proposals_used=observed,
            visual_proposals_seen_fraction=observed / max(1, proposals),
            samples_with_rgb_fraction=samples_with_visual / n,
        )
    return result


def new_model(args, device):
    return FullTokenGeometryLocalizer(
        hidden=args.hidden, text_dim=768, heads=args.heads,
        max_relations=args.max_relations, vision_dim=args.vision_dim,
        top_k=args.top_k, rule_weight=args.rule_weight,
    ).to(device)


def train_one(variant, loaders, encoder, vision, cache, args, device, output):
    model = new_model(args, device)
    if variant == "rgb":
        ckpt = output / "gat_best.pt"
        if not ckpt.exists():
            raise FileNotFoundError("Train gat before rgb; expected gat_best.pt")
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state["model"], strict=True)
        for name, param in model.named_parameters():
            param.requires_grad_(
                name.startswith("vision_proj.") or name.startswith("visual_score.")
            )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr if variant != "rgb" else args.rgb_lr,
        weight_decay=0.01,
    )
    best_seen = -float("inf")
    history = []
    n_epochs = args.rgb_epochs if variant == "rgb" else args.epochs
    for epoch in range(n_epochs):
        model.train()
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        sum_loss, steps, visible = 0.0, 0, 0
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            field = run_model(model, batch, encoder, vision, cache, args, variant, device)
            if variant == "rgb":
                visible += int((field.visual_coverage >= args.min_visual_coverage).sum())
            loss = cross_entropy_goal(
                field.belief_logits.float(),
                batch["target_xy"].to(device), grid_size=args.grid_size,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            sum_loss += float(loss.detach())
            steps += 1
            if args.max_train_batches and steps >= args.max_train_batches:
                break
        if variant == "rgb" and visible == 0:
            raise RuntimeError(
                "Zero observed candidates in RGB train split; A5 cannot learn. "
                "Check observation prefixes, camera footprint and candidate coverage."
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        seen = evaluate(model, loaders["val_seen"], encoder, vision, cache, args, variant, device)
        metric = seen["R@8/20m"]
        log = {
            "variant": variant, "epoch": epoch + 1,
            "loss": sum_loss / max(1, steps),
            "train_seconds": seconds,
            "steps": steps, "valid_visual_candidates_train": visible,
            "peak_gpu_gib": (
                torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                if device.type == "cuda" else None
            ),
            "val_seen": seen,
        }
        history.append(log)
        print(json.dumps(log, ensure_ascii=False), flush=True)
        if metric > best_seen:
            best_seen = metric
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch + 1, "variant": variant,
                "dataset_version": args.dataset_version,
                "match_mode": args.match_mode,
                "selected_on": "val_seen_R@8/20m",
            }, output / f"{variant}_best.pt")
    model.load_state_dict(
        torch.load(output / f"{variant}_best.pt",
                   map_location=device, weights_only=True)["model"]
    )
    return model, history


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    needs_rgb = "rgb" in args.variants
    loaders = {
        "train": make_loader(args.train, args, shuffle=True, visual_required=needs_rgb),
        "val_seen": make_loader(args.val_seen, args, visual_required=needs_rgb),
        "val_unseen": make_loader(args.val_unseen, args, visual_required=needs_rgb),
    }
    query_sets = {k: _query_keys(v.dataset) for k, v in loaders.items()}
    if query_sets["train"] & (query_sets["val_seen"] | query_sets["val_unseen"]):
        raise ValueError("Train/validation static queries overlap: repair splits")
    encoder = FrozenBERTTokens(name=args.text_backbone, device=args.device)
    vision = FrozenSigLIP2(device=args.device) if needs_rgb else None
    cache = CandidateVisionCache(limit=args.visual_cache_size) if needs_rgb else None
    if needs_rgb and args.vision_dim != 768:
        raise ValueError("SigLIP2-base visual embedding dimension is 768")
    report = {
        "protocol": {
            "dataset_version": args.dataset_version, "match_mode": args.match_mode,
            "batch_size": args.batch_size, "hardware": args.hardware,
            "top_k": args.top_k, "grid_size": args.grid_size,
            "rgb_observed_only": True, "checkpoint_selection": "val_seen",
            "evaluation_type": "offline_goal_localization_not_SR_SPL",
        },
        "metrics": {}, "training": {},
    }
    controls = new_model(args, device)
    for variant in BASELINES:
        report["metrics"][variant] = {
            split: evaluate(controls, loaders[split], encoder, vision, cache, args, variant, device)
            for split in ("val_seen", "val_unseen")
        }
    ordered = [x for x in TRAINED if x in args.variants]
    if "rgb" in ordered and "gat" not in ordered and not (output / "gat_best.pt").exists():
        raise ValueError("A5 requires A4 gat_best.pt checkpoint in output directory")
    for variant in ordered:
        model, logs = train_one(
            variant, loaders, encoder, vision, cache, args, device, output,
        )
        report["training"][variant] = logs
        report["metrics"][variant] = {
            split: evaluate(model, loaders[split], encoder, vision, cache, args, variant, device)
            for split in ("val_seen", "val_unseen")
        }
        (output / "full_localization_metrics.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    with (output / "full_localization_metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["variant", "split", "samples", "R@1/20m",
                              "R@4/20m", "R@8/20m", "NE_m"]
        )
        writer.writeheader()
        for variant, split_metrics in report["metrics"].items():
            for split, row in split_metrics.items():
                writer.writerow({
                    "variant": variant, "split": split,
                    **{key: row[key] for key in ("samples", *METRIC_KEYS, "NE_m")}
                })
    print("Saved results to " + str(output / "full_localization_metrics.csv"))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", required=True)
    p.add_argument("--val-seen", required=True)
    p.add_argument("--val-unseen", required=True)
    p.add_argument("--dataset-version", choices=("original", "revised"), required=True)
    p.add_argument("--match-mode", choices=("exact", "typed", "fuzzy"), required=True)
    p.add_argument("--variants", nargs="+", choices=TRAINED, default=list(TRAINED))
    p.add_argument("--roads-json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--hardware", default="RTX Pro 6000 96GB")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--rgb-epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rgb-lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="runs/full_token_rgb")
    p.add_argument("--text-backbone", default="bert-base-uncased")
    p.add_argument("--grid-size", type=int, default=28)
    p.add_argument("--max-landmarks", type=int, default=16)
    p.add_argument("--max-roads", type=int, default=0)
    p.add_argument("--max-relations", type=int, default=8)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--vision-dim", type=int, default=768)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--min-visual-coverage", type=float, default=0.30)
    p.add_argument("--crop-extent-m", type=float, default=100.)
    p.add_argument("--visual-cache-size", type=int, default=4096)
    p.add_argument("--prior-sigma-m", type=float, default=25.)
    p.add_argument("--rule-sigma-m", type=float, default=25.)
    p.add_argument("--rule-weight", type=float, default=1.)
    run(p.parse_args())


if __name__ == "__main__":
    main()
