"""Matched direction-only experiment: text/name vs text/name + public Polygon neighborhood.

No navigation, no RGB, no frame rules, no GT angle at model forward.
Train and evaluate only on TRAIN with held-out maps. Both arms share the
same frozen BERT, target-angle labels, model capacity, optimizer and split.

Example:
python -m scripts.train_polygon_neighborhood_bearing \
  --train-jsonl /path/to/train_revised.jsonl \
  --existing-full-checkpoint /path/to/full_relation/last.pt \
  --public-objects /path/to/cityrefer_original/objects.json \
  --output /path/to/runs/polygon_bearing_s42 --epochs 3
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from graphnav.checkpointing import atomic_json, atomic_save, file_sha256
from graphnav.polygon_neighborhood_bearing import (
    NeighborhoodBearingHead, PublicPolygonNeighborhood,
)
from graphnav.token_language import FrozenBERTTokens
from scripts.train_unified_angle_only import build_data


def choose_maps(records, seed=42, explicit=None):
    maps = sorted({row["map_name"] for row in records})
    if len(maps) < 3:
        raise ValueError("Map-disjoint holdout needs at least three TRAIN maps")
    if explicit:
        held = sorted(set(explicit))
        if not set(held).issubset(maps) or len(held) >= len(maps):
            raise ValueError("Invalid heldout map set")
    else:
        perm = np.random.default_rng(seed).permutation(len(maps))
        held = sorted(maps[int(i)] for i in perm[:max(2, round(len(maps) * .2))])
    return [i for i, row in enumerate(records) if row["map_name"] not in held], [
        i for i, row in enumerate(records) if row["map_name"] in held
    ], held


class BearingExamples(Dataset):
    """Only public reference/neighbor geometry enters inputs; GT is a label."""

    def __init__(self, records, catalog, *, max_refs=16, neighbors=24, radius_m=150.):
        self.records = records
        self.public = PublicPolygonNeighborhood(
            catalog, max_refs=max_refs, max_neighbors=neighbors,
            radius_m=radius_m,
        )
        self.max_refs = max_refs

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        references = row["landmarks"]
        geo = self.public.for_references(row["map_name"], references)
        bounds = np.asarray(row["world_bounds"], np.float32)
        span = bounds[2:] - bounds[:2]
        gt_xy = np.asarray(row["target_xy"], np.float32)
        if gt_xy.shape != (2,) or np.any(span <= 0):
            raise ValueError("Invalid label coordinates or map bounds")
        # GT appears ONLY in this label block, never in PublicPolygonNeighborhood.
        target_world = bounds[:2] + gt_xy * span
        diff = target_world[None, :] - geo["anchor_centers_world"]
        dist = np.linalg.norm(diff, axis=-1)
        theta = np.mod(np.arctan2(diff[:, 1], diff[:, 0]), 2 * np.pi)
        labels = np.rint(theta / (2 * np.pi) * 8).astype(np.int64) % 8
        usable = geo["landmark_valid"] & (dist > 1.)
        labels[~usable] = -100
        angle = theta.astype(np.float32)
        angle[~usable] = np.nan
        names = [(str(x.get("object_type", "")) + " " + str(x["name"])).strip()
                 for x in references]
        names += [""] * (self.max_refs - len(names))
        return dict(index=index, map_name=row["map_name"], text=row["instruction"],
                    names=names, labels=labels, true_angle=angle,
                    n_refs=int(geo["landmark_valid"].sum()),
                    **{key: geo[key] for key in (
                        "anchor_geometry", "neighbor_geometry", "neighbor_types",
                        "neighbor_valid", "landmark_valid"
                    )})


def collate(rows):
    keys = ("anchor_geometry", "neighbor_geometry", "neighbor_types",
            "neighbor_valid", "landmark_valid", "labels", "true_angle")
    batch = {key: torch.from_numpy(np.stack([r[key] for r in rows]))
             for key in keys}
    batch.update(text=[r["text"] for r in rows],
                 names=[r["names"] for r in rows],
                 map_name=[r["map_name"] for r in rows],
                 source_index=[r["index"] for r in rows],
                 n_refs=torch.tensor([r["n_refs"] for r in rows], dtype=torch.long))
    return batch


def predict(head, batch, bert, device, *, geometry_enabled):
    """GT-free forward: receives only explicit public/linguistic input keys."""
    with torch.no_grad():
        tokens, mask = bert.encode_tokens(batch["text"])
        names = bert.encode([name for group in batch["names"] for name in group])
        names = names.reshape(len(batch["text"]), len(batch["names"][0]), -1)
    return head(
        tokens=tokens.detach().to(device), token_mask=mask.to(device),
        node_emb=names.detach().to(device),
        landmark_valid=batch["landmark_valid"].to(device),
        anchor_geometry=batch["anchor_geometry"].to(device),
        neighbor_geometry=batch["neighbor_geometry"].to(device),
        neighbor_types=batch["neighbor_types"].to(device),
        neighbor_valid=batch["neighbor_valid"].to(device),
        geometry_enabled=geometry_enabled,
    )


def angular_loss(logits, labels):
    valid = labels >= 0
    if not valid.any():
        return logits.sum() * 0., 0, 0
    # Equal query weight: multiple public anchors do not multiply query weight.
    ce = F.cross_entropy(logits.transpose(1, 2), labels.clamp_min(0),
                         reduction="none")
    per_query = (ce * valid).sum(1) / valid.sum(1).clamp_min(1)
    chosen = valid.any(1)
    return per_query[chosen].mean(), int(valid.sum()), int(chosen.sum())


@torch.no_grad()
def evaluate(head, dataset, indices, bert, device, *, geometry_enabled,
             batch_size=8, max_batches=0):
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size,
                        shuffle=False, collate_fn=collate, num_workers=0)
    head.eval()
    rows = []
    covered = total_queries = 0
    for step, batch in enumerate(loader):
        if max_batches and step >= max_batches:
            break
        logits = predict(head, batch, bert, device,
                         geometry_enabled=geometry_enabled)
        labels = batch["labels"].to(device)
        angle = batch["true_angle"].to(device)
        valid = labels >= 0
        probs = logits.softmax(-1)
        best = logits.argmax(-1)
        top2 = logits.topk(2, dim=-1).indices
        pred_rad = best.float() * (2 * math.pi / head.bins)
        error = torch.atan2(torch.sin(pred_rad - angle),
                            torch.cos(pred_rad - angle)).abs() * 180 / math.pi
        nll = -probs.gather(-1, labels.clamp_min(0)[..., None]).squeeze(-1).clamp_min(1e-8).log()
        for i in range(len(batch["text"])):
            total_queries += 1
            ids = valid[i].nonzero().flatten().tolist()
            if ids:
                covered += 1
            for j in ids:
                rows.append(dict(index=batch["source_index"][i],
                                 map=batch["map_name"][i],
                                 n_refs=int(batch["n_refs"][i]),
                                 reference_index=j, true_bin=int(labels[i, j]),
                                 predicted_bin=int(best[i, j]),
                                 top1=int(best[i, j] == labels[i, j]),
                                 top2=int((top2[i, j] == labels[i, j]).any()),
                                 error_deg=float(error[i, j]),
                                 nll=float(nll[i, j])))
    def stats(values):
        if not values:
            return dict(n=0, top1=None, top2=None, circular_MAE_deg=None,
                        median_error_deg=None, NLL=None)
        return dict(n=len(values), top1=float(np.mean([r["top1"] for r in values])),
                    top2=float(np.mean([r["top2"] for r in values])),
                    circular_MAE_deg=float(np.mean([r["error_deg"] for r in values])),
                    median_error_deg=float(np.median([r["error_deg"] for r in values])),
                    NLL=float(np.mean([r["nll"] for r in values])))
    return dict(all=stats(rows),
                single_polygon=stats([r for r in rows if r["n_refs"] == 1]),
                multi_polygon=stats([r for r in rows if r["n_refs"] > 1]),
                supported_queries=covered, evaluated_queries=total_queries,
                by_map={m:stats([r for r in rows if r["map"] == m])
                        for m in sorted({r["map"] for r in rows})}), rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--existing-full-checkpoint", type=Path, required=True)
    parser.add_argument("--public-objects", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-count", type=int, default=21878)
    parser.add_argument("--radius-m", type=float, default=150.)
    parser.add_argument("--neighbors", type=int, default=24)
    parser.add_argument("--heldout-maps", nargs="*", default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    args = parser.parse_args()
    if args.seed != 42 or args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Only seed=42, positive batch size and epochs supported")
    if args.output.exists():
        raise FileExistsError("Do not overwrite an existing experiment")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu only for CPU smoke")
    args.output.mkdir(parents=True)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    device = torch.device(args.device)
    source = torch.load(args.existing_full_checkpoint, weights_only=True,
                        map_location="cpu")
    if not source.get("config", {}).get("geometry_args"):
        raise ValueError("Expected existing Unified full relation checkpoint")
    ds = build_data(args.train_jsonl, source,
                    expected=args.expected_count, public_objects=args.public_objects)
    catalog = json.loads(args.public_objects.read_text(encoding="utf8"))
    cfg = source["config"]["geometry_args"]
    dataset = BearingExamples(ds.records, catalog,
                              max_refs=int(cfg["max_landmarks"]),
                              neighbors=args.neighbors, radius_m=args.radius_m)
    train_ids, val_ids, held_maps = choose_maps(ds.records, explicit=args.heldout_maps)
    bert = FrozenBERTTokens(cfg["text_backbone"], device=device,
                            max_length=int(cfg["max_text_length"]))
    original_cache = args.existing_full_checkpoint.parent / "bert_cache.pt"
    if original_cache.exists():
        bert.load_cache(original_cache)
    config = dict(format="polygon-neighborhood-bearing-v1", seed=args.seed,
                  n_train=len(train_ids), n_holdout=len(val_ids),
                  heldout_maps=held_maps, epochs=args.epochs, batch=args.batch_size,
                  lr=args.lr, radius_m=args.radius_m, neighbors=args.neighbors,
                  expected_count=args.expected_count,
                  max_train_batches=args.max_train_batches,
                  max_eval_batches=args.max_eval_batches,
                  train_sha256=file_sha256(args.train_jsonl),
                  catalog_sha256=file_sha256(args.public_objects),
                  source_sha256=file_sha256(args.existing_full_checkpoint),
                  supervised_target_angle_only_in_loss=True,
                  no_rgb_no_reference_frame_no_navigation=True)
    atomic_json(config, args.output / "protocol.json")
    results = {}
    for name, use_geometry in (("text_name_control", False),
                               ("public_polygon_neighborhood", True)):
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        head = NeighborhoodBearingHead().to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr,
                                      weight_decay=.01)
        history = []
        for epoch in range(args.epochs):
            order = np.random.default_rng(42 + epoch).permutation(train_ids).tolist()
            loader = DataLoader(Subset(dataset, order), batch_size=args.batch_size,
                                collate_fn=collate, num_workers=0)
            head.train()
            loss_total = labeled_total = queries_total = batches = 0
            for step, batch in enumerate(loader):
                if args.max_train_batches and step >= args.max_train_batches:
                    break
                logits = predict(head, batch, bert, device,
                                 geometry_enabled=use_geometry)
                loss, labeled, queries = angular_loss(
                    logits, batch["labels"].to(device))
                if labeled == 0:
                    continue
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite bearing loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
                optimizer.step()
                loss_total += float(loss.detach()) * queries
                labeled_total += labeled
                queries_total += queries
                batches += 1
            if batches == 0:
                raise RuntimeError("No valid GT angle supervision in TRAIN")
            entry = dict(epoch=epoch+1, mean_loss=loss_total/max(queries_total, 1),
                         supervised_anchors=labeled_total,
                         supervised_queries=queries_total)
            history.append(entry)
            atomic_save(dict(format="polygon-neighborhood-bearing-v1",
                             arm=name, model=head.state_dict(),
                             epoch=epoch+1, protocol=config,
                             trained_on_heldout=False),
                        args.output / (name + "_last.pt"))
            atomic_json(history, args.output / (name + "_history.json"))
            print(json.dumps(dict(arm=name, **entry)), flush=True)
        report, rows = evaluate(head, dataset, val_ids, bert, device,
                                geometry_enabled=use_geometry,
                                batch_size=args.batch_size,
                                max_batches=args.max_eval_batches)
        results[name] = report
        with (args.output / (name + "_heldout_rows.jsonl")).open("w") as out:
            for item in rows:
                out.write(json.dumps(item) + "\n")
    summary = dict(protocol=config, arms=results,
                   comparison="matched architecture, data, split, seed and optimizer",
                   NOT_a_navigation_result=True,
                   not_an_independent_external_dataset=True,
                   supervised_direction_is_geometric_GT_not_proven_language_semantics=True)
    atomic_json(summary, args.output / "REPORT.json")
    bert.save_cache(args.output / "bert_cache.pt")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
