"""Train GraphNav without HETT checkpoints or networks.

Synthetic smoke mode is intentionally self-contained. Real training uses
CityNav JSONL converted with scripts/prepare_citynav.py and frozen BERT.
This is supervised open-loop training, NOT a closed-loop UAV benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from .config import GraphNavConfig
from .data import CityNavJSONL, collate_episodes, encode_batch_text
from .losses import navigation_losses
from .model import GraphNav


def synthetic_minibatch(config, batch_size, device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    b, m, p = batch_size, config.max_landmarks, config.contour_points
    hidden_text = config.text_dim
    theta = torch.arange(p) * (2 * torch.pi / p)
    circle = torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)
    centers = 0.2 + torch.rand((b, m, 1, 2), generator=g) * 0.6
    poly = centers + circle.view(1, 1, p, 2) * 0.02
    goal = (centers[:, 0, 0] + torch.tensor([0.08, 0.06])).clamp(0, 1)
    start = (centers[:, 0, 0] - torch.tensor([0.10, 0.05])).clamp(0, 1)
    uav = torch.cat((start, torch.zeros(b, 1), torch.ones(b, 1)), -1)
    time = torch.linspace(1.0 / config.trajectory_waypoints, 1.0,
                          config.trajectory_waypoints).reshape(1, -1, 1)
    path = start[:, None] + time * (goal - start)[:, None]
    # Fixed samples' text differs by identity, and is observable at inference.
    features = torch.randn((b, hidden_text), generator=g)
    names = torch.randn((b, m, hidden_text), generator=g)
    mask = torch.zeros((b, m), dtype=torch.bool)
    mask[:, :2] = True
    model_input = dict(
        instruction=features.to(device), landmark_names=names.to(device),
        landmark_contours=poly.to(device), landmark_valid=mask.to(device),
        uav_pose=uav.to(device),
        spatial_maps=torch.zeros(b, 2, config.grid_size, config.grid_size, device=device),
        physical_map_sizes=torch.full((b, 2), 400.0, device=device),
    )
    labels = dict(
        target_xy=goal.to(device), teacher_future=path.to(device),
        physical_map_sizes=torch.full((b, 2), 400.0, device=device),
    )
    return model_input, labels


def init_distributed(device_argument):
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return False, 0, torch.device(device_argument)
    import torch.distributed as dist

    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return True, local_rank, device


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    distributed, local_rank, device = init_distributed(args.device)
    c = GraphNavConfig(
        grid_size=args.grid_size, max_landmarks=args.max_landmarks,
        contour_points=args.contour_points,
        goal_top_k=args.top_k, hidden_dim=args.hidden,
        trajectory_waypoints=args.waypoints,
        text_dim=args.text_dim,
    )
    model = GraphNav(c).to(device)
    encoder = None
    loader = None
    sampler = None
    if args.use_vision and args.smoke:
        raise ValueError('SigLIP2 training requires real observed RGB frames')
    vision_encoder = None
    vision_cache = None
    if not args.smoke:
        if not args.data:
            raise ValueError("use --data prepared.jsonl or --smoke")
        from .foundations import FrozenBERT

        encoder = FrozenBERT(name=args.text_backbone, device=str(device))
        if c.text_dim != 768:
            raise ValueError("BERT features are 768-dimensional")
        if args.use_vision:
            from .foundations import FrozenSigLIP2, CandidateVisionCache
            vision_encoder = FrozenSigLIP2(device=str(device))
            vision_cache = CandidateVisionCache()
        dataset = CityNavJSONL(args.data, c)
        sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
        loader = DataLoader(
            dataset, batch_size=args.batch_size, num_workers=args.workers,
            sampler=sampler, shuffle=(sampler is None),
            pin_memory=(device.type == "cuda"), collate_fn=collate_episodes,
        )
    if distributed:
        from torch.nn.parallel import DistributedDataParallel

        model = DistributedDataParallel(
            model, device_ids=[local_rank] if device.type == "cuda" else None
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    use_amp = device.type == "cuda"
    all_metrics = []
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        batches = range(args.steps) if args.smoke else loader
        epoch_losses = []
        for index, batch in enumerate(batches):
            if args.smoke:
                inputs, target = synthetic_minibatch(
                    c, args.batch_size, device, args.seed + index
                )
            else:
                inputs, target = encode_batch_text(batch, encoder, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                out = model(**inputs)
                if args.use_vision:
                    from .visual_supervision import visual_candidate_batch
                    features, visual_coverage = visual_candidate_batch(
                        out, batch["rgb_observations"], batch["world_bounds"],
                        vision_encoder, vision_dim=c.vision_dim,
                        min_coverage=c.min_visual_coverage, cache=vision_cache,
                    )
                    base = model.module if distributed else model
                    out = base.apply_visual_evidence(out, features, visual_coverage)
                losses = navigation_losses(
                    out, target["target_xy"], inputs["uav_pose"], c,
                    teacher_future=target["teacher_future"],
                    physical_map_sizes=target["physical_map_sizes"],
                    trajectory_weight=0 if epoch < args.goal_warmup_epochs else 0.5,
                )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(float(losses["total"].detach()))
        metric = {"epoch": epoch, "steps": len(epoch_losses),
                  "loss_mean": float(np.mean(epoch_losses))}
        all_metrics.append(metric)
        if local_rank == 0:
            print(json.dumps(metric))
    if local_rank == 0:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        module = model.module if distributed else model
        torch.save(
            {"model": module.state_dict(), "config": c.to_dict(),
             "epochs": args.epochs, "training_metrics": all_metrics},
            path,
        )
        path.with_suffix(".metrics.json").write_text(
            json.dumps(all_metrics, indent=2), encoding="utf-8"
        )
        print(f"Saved checkpoint to {path}")
    if distributed:
        import torch.distributed as dist
        dist.destroy_process_group()
    return all_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--use-vision", action="store_true",
                   help="Train SigLIP2 visual residual from causal observed RGB prefixes")
    p.add_argument("--data", default=None)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--steps", type=int, default=2, help="steps in synthetic smoke mode")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--goal-warmup-epochs", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default="runs/graphnav/checkpoint.pt")
    p.add_argument("--text-backbone", default="bert-base-uncased")
    p.add_argument("--grid-size", type=int, default=28)
    p.add_argument("--max-landmarks", type=int, default=16)
    p.add_argument("--contour-points", type=int, default=16)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--text-dim", type=int, default=768)
    p.add_argument("--waypoints", type=int, default=8)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
