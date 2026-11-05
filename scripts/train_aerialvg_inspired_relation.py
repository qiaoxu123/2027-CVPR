"""TRAIN-only CityNav pilot of an independent AerialVG-inspired relation head.

Frozen original UnifiedPGVB checkpoint/Geometry/PGVB/Stop. Optimizes ONLY the
new candidate-anchor pairwise head. Reuses existing 573-task direct-target
inventory and static cache, not the full CityNav training corpus.

Example (run from GraphNav repository root):
  python scripts/train_aerialvg_inspired_relation.py \\
    --inventory <TRAIN_inventory> --cache <TRAIN_cache> \\
    --checkpoint <trained_direct_target_checkpoint> \\
    --output runs/aerialvg_inspired_train_s42
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from graphnav.aerialvg_inspired_relation import (
    CandidateAnchorRelation, apply_relation_residual,
)
from graphnav.checkpointing import file_sha256
from graphnav.direct_target_data import batch_inputs, load_model


def load_rows(inventory, cache):
    tasks = json.loads((inventory / "public_tasks.json").read_text())
    labels = json.loads((inventory / "private_labels.json").read_text())
    rows = {"fit": [], "dev": []}
    for task in tasks:
        if task["subset"] not in rows:
            continue
        item = torch.load(
            cache / f"{task['index']:05d}.pt",
            map_location="cpu", weights_only=True,
        )
        if item["index"] != task["index"] or item["subset"] != task["subset"]:
            raise ValueError("Cache and TRAIN task inventory disagree")
        # Static map/text query is the SAME across three sampled frames.
        # No RGB or target-dependent crop enters this relation-only pilot.
        row = dict(query=item["query"], task=task,
                   label=labels[task["index"]]["target_world"])
        rows[task["subset"]].append(row)
    if not rows["fit"] or not rows["dev"]:
        raise ValueError("Expected nonempty TRAIN fit/dev splits")
    return rows


def distance_to_target(rows, static_inputs, device):
    bounds = torch.tensor(
        [x["task"]["world_bounds"] for x in rows], device=device,
        dtype=torch.float32,
    )
    world = bounds[:, None, :2] + static_inputs["grid_xy"].float() * (
        bounds[:, None, 2:] - bounds[:, None, :2]
    )
    labels = torch.tensor([x["label"] for x in rows],
                          device=device, dtype=torch.float32)
    return torch.linalg.vector_norm(world - labels[:, None], dim=-1)


def fields(model, head, rows, device):
    inputs = batch_inputs(rows, device=device, visual=False)
    # Never cache the geometry field across distinct query batches.
    model.reset_episode()
    with torch.no_grad():
        base = model._geometry_field(
            inputs["static_inputs"], inputs["tokens"],
            inputs["token_mask"], inputs["node_emb"],
        )
    logits = base.belief_logits.float().detach()
    dense = base.dense_context.float().detach()
    delta = head(
        dense, inputs["tokens"], inputs["token_mask"],
        inputs["relation_geometry"], inputs["relation_ids"],
        inputs["relation_valid"],
    )
    return logits, apply_relation_residual(logits, delta), inputs


@torch.no_grad()
def evaluate(model, head, rows, batch_size, device):
    head.eval()
    results = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i+batch_size]
        original, learned, inputs = fields(model, head, chunk, device)
        dist = distance_to_target(chunk, inputs["static_inputs"], device)
        for j, row in enumerate(chunk):
            a = int(original[j].argmax())
            b = int(learned[j].argmax())
            results.append(dict(
                index=row["task"]["index"],
                original_NE=float(dist[j, a]),
                learned_NE=float(dist[j, b]),
                original_R20=bool(dist[j, a] <= 20),
                learned_R20=bool(dist[j, b] <= 20),
            ))
    n = len(results)
    metrics = dict(
        n=n,
        Geometry_R20=sum(x["original_R20"] for x in results) / n,
        Relation_R20=sum(x["learned_R20"] for x in results) / n,
        Geometry_NE=float(np.mean([x["original_NE"] for x in results])),
        Relation_NE=float(np.mean([x["learned_NE"] for x in results])),
        fixes=sum(not x["original_R20"] and x["learned_R20"] for x in results),
        breaks=sum(x["original_R20"] and not x["learned_R20"] for x in results),
    )
    return metrics, results


def main():
    p = argparse.ArgumentParser()
    for k in ("inventory", "cache", "checkpoint", "output"):
        p.add_argument("--"+k, type=Path, required=True)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        raise ValueError("Invalid training configuration")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for the CityNav geometry checkpoint")

    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = "cuda"
    args.output.mkdir(parents=True, exist_ok=True)
    samples = load_rows(args.inventory, args.cache)
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = load_model(source, device=device).eval().requires_grad_(False)
    head = CandidateAnchorRelation().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.001)
    started = time.perf_counter()
    losses = []
    # Preset epoch count, with no dev-based tuning or early selection.
    for epoch in range(args.epochs):
        head.train()
        random.shuffle(samples["fit"])
        epoch_losses = []
        for start in range(0, len(samples["fit"]), args.batch_size):
            rows = samples["fit"][start:start+args.batch_size]
            original, learned, inputs = fields(model, head, rows, device)
            distance = distance_to_target(rows, inputs["static_inputs"], device)
            teacher = (-distance.square() / (2.0 * 15.0**2)).softmax(-1)
            location = -(teacher * F.log_softmax(learned, -1)).sum(-1).mean()
            preserve = F.kl_div(
                F.log_softmax(learned, -1),
                F.softmax(original, -1),
                reduction="batchmean",
            )
            loss = location + 0.05 * preserve
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        stats = dict(epoch=epoch+1, mean_loss=float(np.mean(epoch_losses)),
                     wall_seconds=time.perf_counter()-started)
        losses.append(stats)
        print(json.dumps(stats), flush=True)

    metrics, cases = evaluate(model, head, samples["dev"], args.batch_size, device)
    payload = dict(
        label="TRAIN-only original-relation pilot; NOT CityNav full navigation",
        seed=args.seed, epochs=args.epochs, train_queries=len(samples["fit"]),
        dev_queries=len(samples["dev"]), metrics=metrics, losses=losses,
        source_sha256=file_sha256(args.checkpoint),
        inventory_sha256=file_sha256(args.inventory/"public_tasks.json"),
        normal_runtime_unchanged=True, Val_Test_accessed=False,
        independent_AerialVG_inspired_implementation=True,
        AerialVG_checkpoint_loaded=False,
    )
    torch.save(dict(format="citynav-aerialvg-inspired-relation-v1",
                    head=head.state_dict(), metadata=payload),
               args.output/"last.pt")
    (args.output/"summary.json").write_text(json.dumps(payload, indent=2))
    (args.output/"dev_cases.json").write_text(json.dumps(cases, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
