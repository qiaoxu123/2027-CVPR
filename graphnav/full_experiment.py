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
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .candidate_geometry import FullCandidateDataset, collate_full
from .full_localization import FullTokenGeometryLocalizer, GoalField
from .token_language import FrozenBERTTokens
from .static_localization import cross_entropy_goal, static_recall
from .foundations import FrozenSigLIP2, CandidateVisionCache
from .visual_supervision import visual_candidate_batch


TRAINED = ("tokens", "geometry", "gat", "rgb")
BASELINES = ("building_prior", "prior", "rules")
METRIC_KEYS = ("R@1/20m", "R@4/20m", "R@8/20m")


def _query_keys(dataset):
    return {(x.get("map_name", ""), x["instruction"]) for x in dataset.records}


def static_signature(dataset):
    """Paired A4/A5 must have identical static inputs and labels, in order."""
    keys = ("map_name", "instruction", "world_bounds", "landmarks", "target_xy",
            "physical_map_sizes", "dataset_version", "match_mode")
    payload = [{k: r.get(k) for k in keys} for r in dataset.records]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def make_loader(path, args, *, shuffle=False, visual_required=False):
    manifest = Path(str(path) + ".provenance.json")
    if manifest.exists():
        provenance = json.loads(manifest.read_text())
        if provenance.get("output_sha256") != hashlib.sha256(Path(path).read_bytes()).hexdigest():
            raise ValueError("Prepared JSONL SHA256 does not match provenance")
        if provenance.get("dataset_version_declared") != args.dataset_version:
            raise ValueError("Source provenance declares a different CityNav version")
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
        persistent_workers=args.workers > 0,
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
    frozen_cache = getattr(model, "_frozen_a4_cache", None) if variant == "rgb" else None
    keys = list(zip(batch["episode_id"], batch["text"]))
    cached = frozen_cache is not None and all(key in frozen_cache for key in keys)
    if not cached:
        observable, tokens, mask, nodes = _batch_inputs(batch, encoder, device, variant)
    with torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")
    ):
        names = ("belief_logits", "candidate_ids", "candidate_xy", "candidate_context")
        if cached:
            field = GoalField(**{name: torch.stack([frozen_cache[key][name] for key in keys]).to(device)
                                 for name in names})
        else:
            field = model(observable, variant=variant, tokens=tokens, token_mask=mask, node_emb=nodes)
            if frozen_cache is not None:
                for i, key in enumerate(keys):
                    frozen_cache[key] = {name: getattr(field, name)[i].detach().cpu().clone()
                                         for name in names}
        if variant == "rgb":
            visual_cache = getattr(model, "_frozen_visual_cache", None)
            if visual_cache is not None and all(key in visual_cache for key in keys):
                embeddings = torch.stack([visual_cache[key][0] for key in keys]).to(device)
                confidence = torch.stack([visual_cache[key][1] for key in keys]).to(device)
            else:
                embeddings, confidence = visual_candidate_batch(
                    field, batch["rgb_observations"], batch["world_bounds"],
                    vision, vision_dim=args.vision_dim,
                    min_coverage=args.min_visual_coverage,
                    crop_extent_m=args.crop_extent_m, cache=image_cache)
                if visual_cache is not None:
                    for i, key in enumerate(keys):
                        visual_cache[key] = (embeddings[i].detach().cpu().clone(),
                                             confidence[i].detach().cpu().clone())
            # Only trusted evidence from the observed causal mosaic enters logits.
            field = model.refine_rgb(
                field, embeddings.to(device), confidence.to(device),
                threshold=args.min_visual_coverage,
            )
    return field


@torch.inference_mode()
def evaluate(model, loader, encoder, vision, cache, args, variant, device, prediction_path=None):
    model.eval()
    total = {"count": 0, "NE_sum_m": 0.,
             **{f"R@{k}/20m_count": 0 for k in (1, 4, 8)}}
    observed, proposals, samples_with_visual, bound, relation = 0, 0, 0, 0, 0
    nonbuilding, road_surface = 0, 0
    predictions = []
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
        if prediction_path is not None:
            # Match static_recall exactly: topk(20) can order tied scores
            # differently from topk(8), so its first eight are not equivalent.
            logits = field.belief_logits.float()
            primary_ids = logits.topk(min(8, logits.shape[-1]), dim=1).indices
            primary_xy = batch["grid_xy"].to(device).gather(
                1, primary_ids[..., None].expand(-1, -1, 2))
            primary_distances = ((primary_xy - batch["target_xy"].to(device)[:, None]) *
                                 batch["map_sizes"].to(device)[:, None]).norm(dim=-1).cpu().tolist()
            ids = logits.topk(min(20, logits.shape[-1]), dim=1).indices
            xy = batch["grid_xy"].to(device).gather(1, ids[..., None].expand(-1, -1, 2))
            distances = ((xy - batch["target_xy"].to(device)[:, None]) *
                         batch["map_sizes"].to(device)[:, None]).norm(dim=-1).cpu().tolist()
            coverage = (field.visual_coverage.cpu().tolist()
                        if field.visual_coverage is not None else [None] * len(distances))
            for i, d in enumerate(distances):
                primary = primary_distances[i]
                predictions.append({"episode_id": batch["episode_id"][i],
                    "instruction": batch["text"][i], "NE_m": primary[0],
                    "hits": {**{str(k): min(primary[:k]) <= 20 for k in (1, 4, 8)},
                             "20": min(d) <= 20},
                    "top8_distances_m": primary,
                    "top20_distances_m": d, "visual_coverage": coverage[i],
                    "relation_count": int(batch["relation_count"][i]),
                    "surface_count": int(batch["surface_count"][i])})
        bound += int((batch["surface_count"] > 0).sum())
        relation += int((batch["relation_count"] > 0).sum())
        nonbuilding += int((batch["nonbuilding_surface_count"] > 0).sum())
        road_surface += int(((batch["node_types"] == 3) & batch["node_valid"]).any(1).sum())
        if field.visual_coverage is not None:
            mask = field.visual_coverage >= args.min_visual_coverage
            observed += int(mask.sum())
            samples_with_visual += int(mask.any(dim=1).sum())
            proposals += int(mask.numel())
    n = max(1, total["count"])
    result = {
        "samples": total["count"],
        "NE_m": total["NE_sum_m"] / n,
        **{f"R@{k}/20m": total[f"R@{k}/20m_count"] / n for k in (1, 4, 8)},
        "surface_anchor_fraction": bound / n,
        "relation_grounded_fraction": relation / n,
        "nonbuilding_surface_fraction": nonbuilding / n,
        "road_surface_fraction": road_surface / n,
    }
    if variant == "rgb":
        result.update(
            visual_proposals_used=observed,
            visual_proposals_seen_fraction=observed / max(1, proposals),
            samples_with_rgb_fraction=samples_with_visual / n,
        )
    if prediction_path is not None:
        Path(prediction_path).write_text("".join(json.dumps(r) + "\n" for r in predictions))
    return result


def new_model(args, device):
    return FullTokenGeometryLocalizer(
        hidden=args.hidden, text_dim=768, heads=args.heads,
        max_relations=args.max_relations, vision_dim=args.vision_dim,
        top_k=args.top_k, rule_weight=args.rule_weight,
        relative_position=getattr(args, "relative_position", False),
    ).to(device)


def train_one(variant, loaders, encoder, vision, cache, args, device, output):
    torch.manual_seed(getattr(args, "seed", 0))
    signatures = {split: static_signature(loader.dataset) for split, loader in loaders.items()}
    model = new_model(args, device)
    pretrained = getattr(args, "init_bidirectional", None)
    if pretrained and variant == "geometry":
        state = torch.load(pretrained, map_location=device, weights_only=True)
        if state.get("training_stage") != "P1_bidirectional_geometry":
            raise ValueError("Expected a verified P1 bidirectional pretrain checkpoint")
        for name in ("hidden", "heads", "grid_size"):
            if state.get(name) != getattr(args, name):
                raise ValueError(f"Bidirectional pretrain {name} incompatible with A3")
        if not getattr(args, "relative_position", False):
            raise ValueError("P1 checkpoint requires --relative-position in A3 finetuning")
        model.load_state_dict(state["model"], strict=True)
    if variant == "rgb":
        ckpt = output / "gat_best.pt"
        if not ckpt.exists():
            raise FileNotFoundError("Train gat before rgb; expected gat_best.pt")
        state = torch.load(ckpt, map_location=device, weights_only=True)
        if state.get("static_signatures") != signatures:
            raise ValueError("A4/A5 static queries, labels or map geometry differ")
        for key in ("hidden", "heads", "max_relations", "grid_size", "top_k", "rule_weight",
                    "max_landmarks", "max_roads", "prior_sigma_m", "rule_sigma_m", "text_backbone"):
            if state.get("args", {}).get(key) != getattr(args, key, None):
                raise ValueError(f"A4/A5 configuration mismatch: {key}")
        if state.get("args", {}).get("max_text_length", 96) != getattr(args, "max_text_length", 96):
            raise ValueError("A4/A5 text truncation policy differs")
        model.load_state_dict(state["model"], strict=True)
        for name, param in model.named_parameters():
            param.requires_grad_(
                name.startswith("vision_proj.") or name.startswith("visual_score.")
            )
        # A4 is immutable and deterministic; cache only its observable output,
        # never labels or a visual refinement. ~200 MB for the full split.
        model._frozen_a4_cache = {}
        model._frozen_visual_cache = {}
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr if variant != "rgb" else args.rgb_lr,
        weight_decay=0.01,
    )
    best_seen = -float("inf")
    history = []
    n_epochs = args.rgb_epochs if variant == "rgb" else args.epochs
    for epoch in range(n_epochs):
        if variant == "rgb":
            # Frozen A4 must remain deterministic (no GAT/attention dropout).
            # Only the small visual adapter receives training-time behavior.
            model.eval()
            model.vision_proj.train()
            model.visual_score.train()
        else:
            model.train()
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        sum_loss, steps, visible, samples = 0.0, 0, 0, 0
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
            count = len(batch["text"])
            sum_loss += float(loss.detach()) * count
            samples += count
            steps += 1
            if variant == "rgb" and steps % 200 == 0:
                print(json.dumps({"variant": variant, "epoch": epoch + 1,
                    "steps": steps, "samples": samples, "observed_proposals": visible,
                    "elapsed_seconds": time.perf_counter() - started}), flush=True)
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
        peak_gpu = (torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                    if device.type == "cuda" else None)
        seen = evaluate(model, loaders["val_seen"], encoder, vision, cache, args, variant, device)
        metric = seen["R@8/20m"]
        log = {
            "variant": variant, "epoch": epoch + 1,
            "loss": sum_loss / max(1, samples),
            "train_seconds": seconds,
            "train_samples": samples, "samples_per_second": samples / max(seconds, 1e-9),
            "steps": steps, "valid_visual_candidates_train": visible,
            "peak_gpu_gib": peak_gpu,
            "val_seen": seen,
        }
        history.append(log)
        (output / f"{variant}_history.json").write_text(json.dumps(history, indent=2))
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1, "variant": variant, "args": vars(args),
                    "history": history}, output / f"{variant}_last.pt")
        print(json.dumps(log, ensure_ascii=False), flush=True)
        if metric > best_seen:
            best_seen = metric
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch + 1, "variant": variant,
                "dataset_version": args.dataset_version,
                "match_mode": args.match_mode,
                "selected_on": "val_seen_R@8/20m",
                "args": vars(args),
                "static_signatures": signatures,
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
    encoder.max_length = getattr(args, "max_text_length", 512)
    text_cache_path = output / f"frozen_tokens_{encoder.max_length}.pt"
    if getattr(args, "evaluate_only", False) and text_cache_path.exists():
        encoder.load_cache(text_cache_path)
    encoders = {(args.text_backbone, encoder.max_length): encoder}
    vision = FrozenSigLIP2(device=args.device) if needs_rgb else None
    cache = CandidateVisionCache(limit=args.visual_cache_size) if needs_rgb else None
    vision_cache_path = output / "frozen_siglip2_features.pt"
    vision_metadata = ({"name": "google/siglip2-base-patch16-224",
        "revision": getattr(vision.model.config, "_commit_hash", None),
        "precision": "bf16" if device.type == "cuda" else "fp32"} if needs_rgb else None)
    if needs_rgb and getattr(args, "evaluate_only", False) and vision_cache_path.exists():
        cache.load(vision_cache_path, vision_metadata)
    if needs_rgb and args.vision_dim != 768:
        raise ValueError("SigLIP2-base visual embedding dimension is 768")
    report = {
        "protocol": {
            "dataset_version": args.dataset_version, "match_mode": args.match_mode,
            "batch_size": args.batch_size, "hardware": args.hardware,
            "top_k": args.top_k, "grid_size": args.grid_size,
            "rgb_observed_only": True, "checkpoint_selection": "val_seen",
            "evaluation_type": "offline_goal_localization_not_SR_SPL",
            "args": vars(args),
            "static_signatures": {k: static_signature(v.dataset) for k, v in loaders.items()},
        },
        "metrics": {}, "training": {},
    }
    controls = new_model(args, device)
    for variant in BASELINES:
        report["metrics"][variant] = {
            split: evaluate(controls, loaders[split], encoder, vision, cache, args, variant, device,
                            output / f"{variant}_{split}_queries.jsonl")
            for split in ("val_seen", "val_unseen")
        }
        print(json.dumps({"baseline": variant, "metrics": report["metrics"][variant]}), flush=True)
        (output / "full_localization_metrics.json").write_text(json.dumps(report, indent=2))
    ordered = [x for x in TRAINED if x in args.variants]
    if "rgb" in ordered and "gat" not in ordered and not (output / "gat_best.pt").exists():
        raise ValueError("A5 requires A4 gat_best.pt checkpoint in output directory")
    for variant in ordered:
        eval_encoder = encoder
        if getattr(args, "evaluate_only", False):
            model = new_model(args, device)
            state = torch.load(output / f"{variant}_best.pt", map_location=device, weights_only=True)
            if state.get("static_signatures") != report["protocol"]["static_signatures"]:
                raise ValueError("Evaluation inputs differ from checkpoint static protocol")
            if bool(state.get("args", {}).get("relative_position", False)) != bool(
                getattr(args, "relative_position", False)
            ):
                raise ValueError("Evaluate-only relative_position differs from checkpoint")
            model.load_state_dict(state["model"])
            text_name = state.get("args", {}).get("text_backbone", args.text_backbone)
            text_length = state.get("args", {}).get("max_text_length", 96)
            key = (text_name, text_length)
            if key not in encoders:
                encoders[key] = FrozenBERTTokens(name=text_name, device=args.device, max_length=text_length)
                cache_path = output / f"frozen_tokens_{text_length}.pt"
                if cache_path.exists():
                    encoders[key].load_cache(cache_path)
            eval_encoder = encoders[key]
            report["protocol"].setdefault("checkpoint_training_args", {})[variant] = state.get("args", {})
            history_path = output / f"{variant}_history.json"
            logs = json.loads(history_path.read_text()) if history_path.exists() else []
        else:
            model, logs = train_one(variant, loaders, encoder, vision, cache, args, device, output)
        report["training"][variant] = logs
        report["metrics"][variant] = {
            split: evaluate(model, loaders[split], eval_encoder, vision, cache, args, variant, device,
                            output / f"{variant}_{split}_queries.jsonl")
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
    if getattr(args, "evaluate_only", False):
        for (_, length), frozen in encoders.items():
            frozen.save_cache(output / f"frozen_tokens_{length}.pt")
        if needs_rgb:
            cache.save(vision_cache_path, vision_metadata)
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
    p.add_argument("--evaluate-only", action="store_true", help="Load selected checkpoints without retraining")
    p.add_argument("--roads-json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--hardware", default="RTX Pro 6000 96GB")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--relative-position", action="store_true",
                   help="Optional P1 relative-position geometry control")
    p.add_argument("--init-bidirectional", default=None,
                   help="Initialize A3 geometry from separately trained P1 checkpoint")
    p.add_argument("--rgb-epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rgb-lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="runs/full_token_rgb")
    p.add_argument("--text-backbone", default="bert-base-uncased")
    p.add_argument("--max-text-length", type=int, default=512)
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
