"""Train only the opt-in target-evidence adapter from causal, precomputed TRAIN tensors.

The serialized training rows must be produced from *observed* RGB patches with
an audited Train Seen split. The target label is removed from the forward
payload and used only in loss(). No validation split is read by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .evidence_grounding import TargetEvidenceRefiner
from .full_localization import GoalField


def evidence_loss(refiner, row):
    """A3 frozen fields and observed patches; private labels isolated below."""
    necessary = (
        "base_logits", "dense_context", "grid_xy", "relation_fields",
        "text_tokens", "text_mask", "patch_features", "patch_xy",
        "patch_valid", "target_distribution",
    )
    if any(k not in row for k in necessary):
        raise ValueError("missing causal evidence training tensors or label")
    base = row["base_logits"].detach().float()
    dense = row["dense_context"].detach().float()
    grid = row["grid_xy"].float()
    if base.ndim != 2 or dense.shape[:2] != base.shape:
        raise ValueError("base logit/context shape mismatch")
    if grid.shape != (*base.shape, 2):
        raise ValueError("unmatched candidate positions")
    label = row["target_distribution"].float()
    if label.shape != base.shape or not torch.isfinite(label).all() or (label < 0).any():
        raise ValueError("invalid supervised target probability")
    label = label / label.sum(-1, keepdim=True).clamp_min(1e-9)
    if not (label.sum(-1) > 0).all():
        raise ValueError("every training row requires a target label")
    top = base.topk(min(20, base.shape[-1]), -1).indices
    field = GoalField(
        base, top, grid.gather(1, top[..., None].expand(-1, -1, 2)),
        dense.gather(1, top[..., None].expand(-1, -1, dense.shape[-1])),
        dense_context=dense,
    )
    result = refiner(
        field, grid_xy=grid,
        relation_fields=row["relation_fields"],
        text_tokens=row["text_tokens"], text_mask=row["text_mask"],
        patch_features=row["patch_features"],
        patch_xy=row["patch_xy"], patch_valid=row["patch_valid"],
        observed_grid_valid=row.get("observed_grid_valid"),
    )
    goal_ce = -(label * F.log_softmax(result.belief_logits.float(), dim=-1)).sum(-1).mean()
    # A3 preservation regularizer guards against ungrounded arbitrary shifts.
    old = F.softmax(base, dim=-1).detach()
    distill = F.kl_div(
        F.log_softmax(result.belief_logits.float(), dim=-1),
        old, reduction="batchmean",
    )
    return goal_ce + 0.1 * distill, {"ce": float(goal_ce.detach()), "kl": float(distill.detach())}


def load_rows(path):
    dataset = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(dataset, dict) or dataset.get("split") != "train_seen":
        raise ValueError("only audited train_seen evidence data is supported")
    if dataset.get("causal_observations") is not True:
        raise ValueError("observed-only causal supervision manifest required")
    rows = dataset.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("no evidence training rows")
    return rows


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    source = Path(args.rows)
    rows = load_rows(source)
    config = dict(
        hidden=args.hidden, text_dim=args.text_dim,
        vision_dim=args.vision_dim, relation_dim=args.relation_dim,
    )
    model = TargetEvidenceRefiner(**config).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    for epoch in range(args.epochs):
        order = torch.randperm(len(rows)).tolist()
        running = 0.
        for index in order:
            row = {k: (v.to(device) if torch.is_tensor(v) else v)
                   for k, v in rows[index].items()}
            optimizer.zero_grad(set_to_none=True)
            loss, _ = evidence_loss(model, row)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite evidence training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            running += float(loss.detach())
        print(json.dumps(dict(epoch=epoch + 1, train_loss=running / len(order))))
    from .checkpointing import file_sha256
    checkpoint = dict(
        protocol="graphnav-target-evidence-v1",
        model={key: value.detach().cpu() for key, value in model.state_dict().items()},
        config=config,
        a3_sha256=file_sha256(args.a3_checkpoint),
        vision_backbone=args.vision_backbone,
        training_data_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        split="train_seen",
        epochs=args.epochs,
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    from .checkpointing import atomic_save
    atomic_save(checkpoint, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", required=True)
    parser.add_argument("--a3-checkpoint", required=True)
    parser.add_argument("--vision-backbone", default="google/siglip2-large-patch16-384")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--text-dim", type=int, default=768)
    parser.add_argument("--vision-dim", type=int, default=1024)
    parser.add_argument("--relation-dim", type=int, default=8)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
