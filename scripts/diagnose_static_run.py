"""Per-query metrics, paired uncertainty, sigma sensitivity, and topology intervention."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from graphnav.foundations import FrozenBERT
from graphnav.static_experiment import StageSettings, forward_batch, load_data
from graphnav.static_localization import StaticGoalLocalizer


def summarize(distances):
    return {
        "samples": len(distances),
        "NE_m": float(distances[:, 0].mean()),
        **{f"R@{k}/20m": float((distances[:, :k].min(1) <= 20).mean()) for k in (1, 4, 8)},
    }


def paired(a, b, seed=0):
    rng = np.random.default_rng(seed)
    result = {}
    for k in (1, 4, 8):
        delta = (a[:, :k].min(1) <= 20).astype(float) - (b[:, :k].min(1) <= 20).astype(float)
        boot = np.asarray(
            [delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(5000)]
        )
        result[f"R@{k}/20m"] = {
            "delta": float(delta.mean()),
            "paired_bootstrap_95ci": np.quantile(boot, [0.025, 0.975]).tolist(),
            "wins": int((delta > 0).sum()),
            "losses": int((delta < 0).sum()),
        }
    delta = a[:, 0] - b[:, 0]
    boot = np.asarray([delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(5000)])
    result["NE_m"] = {
        "delta": float(delta.mean()),
        "paired_bootstrap_95ci": np.quantile(boot, [0.025, 0.975]).tolist(),
    }
    return result


@torch.inference_mode()
def diagnose(args):
    out = Path(args.output)
    device = torch.device(args.device)
    ckpt = torch.load(out / "best.pt", map_location=device, weights_only=False)
    settings = StageSettings(
        **{
            k: v
            for k, v in ckpt["settings"].items()
            if k in {f.name for f in fields(StageSettings)}
        }
    )
    model = (
        StaticGoalLocalizer(
            hidden=settings.hidden,
            text_dim=settings.text_dim,
            heads=settings.heads,
            relations=settings.max_relations,
            rule_weight=settings.rule_weight,
        )
        .to(device)
        .eval()
    )
    model.load_state_dict(ckpt["model"])
    encoder = FrozenBERT(name=ckpt["text_backbone"], device=args.device)
    feature_path = out / "frozen_text_features.pt"
    if feature_path.exists():
        encoder.cache = {
            text: feature.to(device)
            for text, feature in torch.load(
                feature_path, map_location="cpu", weights_only=True
            ).items()
        }
    results = {
        "selected_epoch": ckpt["epoch"],
        "graph_scale_tanh": float(torch.tanh(model.graph_scale)),
        "protocol": "Post-hoc evaluation only; 5000 paired query bootstrap resamples. Topology intervention uses trained weights and self edges, without retraining.",
        "splits": {},
    }
    for split in ("val_seen", "val_unseen"):
        loader = load_data(Path(args.data_dir) / f"graphnav_{split}.jsonl", settings, None, 16)
        distances = {v: [] for v in ("prior", "rules", "gat", "gat_self_only", "map_center")}
        sigma_dist = {str(s): [] for s in (10, 20, 30, 40)}
        metadata = []
        started = time.perf_counter()
        for batch in loader:

            def measure(logits, current_batch=batch):
                ids = logits.topk(8, dim=-1).indices.cpu()
                xy = current_batch["grid_xy"].gather(1, ids[..., None].expand(-1, -1, 2))
                return (
                    (
                        (xy - current_batch["target_xy"][:, None])
                        * current_batch["map_sizes"][:, None]
                    )
                    .norm(dim=-1)
                    .numpy()
                )

            for variant in ("prior", "rules", "gat"):
                logits, _ = forward_batch(
                    model, encoder, batch, device, variant, device.type == "cuda"
                )
                distances[variant].append(measure(logits))
            distances["map_center"].append(
                measure(
                    -((batch["grid_xy"].to(device) - 0.5) * batch["map_sizes"].to(device)[:, None])
                    .square()
                    .sum(-1)
                )
            )
            for sigma in (10, 20, 30, 40):
                sigma_dist[str(sigma)].append(
                    measure(batch["prior"].to(device) * settings.prior_sigma_m / sigma)
                )
            intervened = dict(batch)
            n = batch["adjacency"].shape[-1]
            intervened["adjacency"] = batch["adjacency"] & torch.eye(n, dtype=torch.bool)[None]
            logits, _ = forward_batch(
                model, encoder, intervened, device, "gat", device.type == "cuda"
            )
            distances["gat_self_only"].append(measure(logits))
            for i, query in enumerate(batch["episode_id"]):
                metadata.append(
                    {
                        "query_id": query,
                        "relation_count": int(batch["relation_count"][i]),
                        "building_count": len(loader.dataset.records[len(metadata)]["landmarks"]),
                        "instruction": batch["text"][i],
                    }
                )
        distances = {k: np.concatenate(v) for k, v in distances.items()}
        sigma_dist = {k: np.concatenate(v) for k, v in sigma_dist.items()}
        masks = {
            "all": np.ones(len(metadata), bool),
            "grounded_relation": np.asarray([m["relation_count"] > 0 for m in metadata]),
            "no_grounded_relation": np.asarray([m["relation_count"] == 0 for m in metadata]),
            # A null node has building type zero; consult source landmarks instead.
            "with_building": np.asarray([bool(r["landmarks"]) for r in loader.dataset.records]),
            "no_building": np.asarray([not r["landmarks"] for r in loader.dataset.records]),
        }
        results["splits"][split] = {
            "metrics": {k: summarize(v) for k, v in distances.items()},
            "sigma_sensitivity_S0": {k: summarize(v) for k, v in sigma_dist.items()},
            "S2_minus_S1": paired(distances["gat"], distances["rules"]),
            "S2_minus_self_only": paired(distances["gat"], distances["gat_self_only"]),
            "subgroups": {
                group: {v: summarize(d[mask]) for v, d in distances.items()}
                for group, mask in masks.items()
                if mask.any()
            },
            "diagnosis_seconds": time.perf_counter() - started,
        }
        np.savez_compressed(out / f"{split}_top8_distances.npz", **distances)
        rows = []
        for i, meta in enumerate(metadata):
            for variant, d in distances.items():
                rows.append(
                    {
                        **meta,
                        "variant": variant,
                        "NE_m": float(d[i, 0]),
                        **{f"R@{k}/20m": int(d[i, :k].min() <= 20) for k in (1, 4, 8)},
                    }
                )
        with (out / f"{split}_per_query.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(split, json.dumps(results["splits"][split]["metrics"]), flush=True)
    (out / "diagnostics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    diagnose(p.parse_args())
