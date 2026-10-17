"""Offline goal-recall diagnostics; NOT closed-loop SR/SPL evaluation."""

from __future__ import annotations

import argparse
import json
import torch
from torch.utils.data import DataLoader

from .config import GraphNavConfig
from .data import CityNavJSONL, collate_episodes, encode_batch_text
from .foundations import FrozenBERT
from .geometry import grid_centers
from .model import GraphNav


def open_loop_metrics(pred, target_xy, physical_map_sizes, grid_size, topks=(1, 5, 20)):
    """Return measured goal distance and top-k hit counters at 20 physical meters."""
    xy = grid_centers(
        grid_size, device=pred.belief_logits.device,
        dtype=pred.belief_logits.dtype,
    )
    order = pred.belief_logits.topk(
        min(max(topks), xy.shape[0]), dim=-1
    ).indices
    positions = xy[order]
    distances = (
        (positions - target_xy[:, None]) * physical_map_sizes[:, None]
    ).norm(dim=-1)
    result = {
        "samples": target_xy.shape[0],
        "NE_top1_sum_m": distances[:, 0].sum().item(),
    }
    for k in topks:
        effective_k = min(k, distances.shape[1])
        result[f"R@{k}/20m_hits"] = (
            distances[:, :effective_k].amin(dim=-1) <= 20
        ).sum().item()
    return result


def run(args):
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = GraphNavConfig(**ckpt["config"])
    device = torch.device(args.device)
    model = GraphNav(config).to(device).eval()
    model.load_state_dict(ckpt["model"], strict=True)
    language = FrozenBERT(name=args.text_backbone, device=args.device)
    dataset = CityNavJSONL(args.data, config)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=collate_episodes,
        num_workers=args.workers,
    )
    counts = {"samples": 0, "NE_top1_sum_m": 0.0}
    for k in (1, 5, 20):
        counts[f"R@{k}/20m_hits"] = 0
    with torch.inference_mode():
        for batch in loader:
            features, target = encode_batch_text(batch, language, device)
            pred = model(**features)
            stats = open_loop_metrics(
                pred, target["target_xy"], target["physical_map_sizes"],
                config.grid_size,
            )
            for key, value in stats.items():
                counts[key] += value
    n = counts["samples"]
    summary = {"samples": n, "NE_top1_m": counts["NE_top1_sum_m"] / n}
    for k in (1, 5, 20):
        summary[f"R@{k}/20m"] = counts[f"R@{k}/20m_hits"] / n
    print(json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--text-backbone", default="bert-base-uncased")
    run(p.parse_args())


if __name__ == "__main__":
    main()
