"""Joint Geometry + AerialVG-inspired relation TRAIN on *all* CityNav queries.

Use static-only, revised TRAIN JSONL produced by scripts/prepare_citynav.py.
Unmatched names/indirect instructions are NOT filtered. Frozen pretrained BERT
is an upstream cached feature source; no unobserved RGB or GT enters forward.
The existing PGVB visual weights remain available in the final integrated
checkpoint but are NOT re-trained by this static-only loss.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from graphnav.candidate_geometry import FullCandidateDataset, collate_full
from graphnav.checkpointing import atomic_json, atomic_save, file_sha256
from graphnav.full_relation_training import (
    FORMAT, full_relation_inputs, load_full_relation_model,
)
from graphnav.static_localization import cross_entropy_goal
from graphnav.token_language import FrozenBERTTokens


def full_loader(path, args, *, shuffle):
    dataset = FullCandidateDataset(
        path, grid_size=28,
        max_landmarks=args.max_landmarks,
        max_roads=args.max_roads,
        max_relations=args.max_relations,
        prior_sigma_m=args.prior_sigma_m,
        rule_sigma_m=args.rule_sigma_m,
        deduplicate_queries=False,
    )
    versions = {q.get("dataset_version") for q in dataset.records}
    matches = {q.get("match_mode") for q in dataset.records}
    if versions != {"revised"} or matches != {"fuzzy"}:
        raise ValueError(f"Full training requires revised/fuzzy, not {versions}/{matches}")
    # Source with step-sampled duplicates would not be a query-complete
    # experiment: require one original instruction/target record per row.
    if any("teacher_future" in q or "uav_pose" in q for q in dataset.records):
        raise ValueError("Prepare the source with --static-only, not teacher trajectory prefixes")
    return dataset, DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle,
        num_workers=0, collate_fn=collate_full,
    )


def paired_metrics(original, refined, batch):
    target = batch["target_xy"].to(refined.device)
    sizes = batch["map_sizes"].to(refined.device)
    grid = batch["grid_xy"].to(refined.device)
    indices = torch.stack((original.argmax(-1), refined.argmax(-1)), -1)
    predictions = grid.gather(1, indices[..., None].expand(-1, -1, 2))
    dist = torch.linalg.vector_norm(
        (predictions - target[:, None]) * sizes[:, None], dim=-1
    )
    return dist.detach().cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    for name in ("train-jsonl", "stage1", "original-v2", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr-geometry", type=float, default=1e-5)
    p.add_argument("--lr-relation", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-batches", type=int, default=0,
                   help="Smoke-only; any value > 0 invalidates full training claim")
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This full-scale training protocol requires CUDA")
    if a.epochs < 1 or a.batch_size < 1 or a.max_batches < 0:
        raise ValueError("Invalid training configuration")
    torch.set_num_threads(4)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False

    a.output.mkdir(parents=True, exist_ok=True)
    dataset, loader = full_loader(a.train_jsonl, None if False else
        argparse.Namespace(**dict(
            max_landmarks=16, max_roads=0, max_relations=8,
            prior_sigma_m=25.0, rule_sigma_m=25.0,
            batch_size=a.batch_size
        )), shuffle=True)

    source_sha = dict(
        stage1=file_sha256(a.stage1),
        original_v2=file_sha256(a.original_v2),
        train_jsonl=file_sha256(a.train_jsonl),
    )
    original = torch.load(a.original_v2, map_location="cpu", weights_only=True)
    if original.get("format") != "unified-pgvb-v2":
        raise ValueError("Requires official integrated UnifiedPGVB-v2")
    if original["config"]["source"]["stage1_sha256"] != source_sha["stage1"]:
        raise ValueError("Original geometry/controller checksum mismatch")
    geometry_args = original["config"]["geometry_args"]
    if int(geometry_args["grid_size"]) != 28:
        raise ValueError("Unexpected controller input grid")
    # Rebuild the dataset using actual Geometry checkpoint feature protocol.
    actual = argparse.Namespace(
        batch_size=a.batch_size,
        max_landmarks=int(geometry_args["max_landmarks"]),
        max_roads=int(geometry_args["max_roads"]),
        max_relations=int(geometry_args["max_relations"]),
        prior_sigma_m=float(geometry_args["prior_sigma_m"]),
        rule_sigma_m=float(geometry_args["rule_sigma_m"]),
    )
    if (actual.max_landmarks, actual.max_roads, actual.max_relations,
            actual.prior_sigma_m, actual.rule_sigma_m) != (16, 0, 8, 25.0, 25.0):
        dataset, loader = full_loader(a.train_jsonl, actual, shuffle=True)
    if len(dataset) < 1:
        raise ValueError("Empty full CityNav TRAIN dataset")
    # Every source row stays in the denominator; no direct-relation parser.
    count = len(dataset.records)
    missing_anchor = sum(not row.get("landmarks") for row in dataset.records)
    manifest = dict(source_sha256=source_sha, records=count,
        no_public_anchor_rows=missing_anchor, source="complete revised CityNav TRAIN",
        eligibility_filter=None, checkpoint_format=FORMAT,
        epochs=a.epochs, batch_size=a.batch_size, seed=a.seed,
        smoke=bool(a.max_batches), max_batches=a.max_batches)
    manifest_path = a.output/"manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Resume inputs/configuration changed")
    if a.resume and not manifest_path.exists():
        raise ValueError("Resume requires existing training manifest")
    if not a.resume and (a.output/"last.pt").exists():
        raise FileExistsError("Training checkpoint exists; use --resume")
    atomic_json(manifest, manifest_path)

    previous = None
    if a.resume:
        previous = torch.load(a.output/"last.pt",
                              map_location="cpu", weights_only=True)
        if previous["source_sha256"] != source_sha:
            raise ValueError("Resume source fingerprint mismatch")
        model = load_full_relation_model(previous, "cuda")
        start_epoch = int(previous["completed_epochs"])
    else:
        model = load_full_relation_model(original, "cuda")
        start_epoch = 0
    model.requires_grad_(False)
    model.geometry.requires_grad_(True)
    model.full_relation.requires_grad_(True)
    # PGVB has no supervision from a static-only sample; keep it at verified
    # pretrained weights rather than silently training against fake pixels.
    optimizer = torch.optim.AdamW([
        dict(params=list(model.geometry.parameters()), lr=a.lr_geometry),
        dict(params=list(model.full_relation.parameters()), lr=a.lr_relation),
    ], weight_decay=0.001)
    if previous is not None:
        optimizer.load_state_dict(previous["optimizer"])
        torch.set_rng_state(previous["torch_rng"])
        torch.cuda.set_rng_state_all(previous["cuda_rng"])
        random.setstate(previous["python_rng"])
        np.random.set_state(previous["numpy_rng"])
    bert = FrozenBERTTokens(
        geometry_args["text_backbone"], device="cuda",
        max_length=int(geometry_args["max_text_length"]),
    )
    history = previous["history"] if previous is not None else []
    started = time.perf_counter()
    for epoch in range(start_epoch, a.epochs):
        model.train()
        loss_total = 0.0
        seen = 0
        corrected = damaged = 0
        for index, batch in enumerate(loader):
            if a.max_batches and index >= a.max_batches:
                break
            inputs = full_relation_inputs(batch, bert, "cuda")
            with torch.autocast("cuda", enabled=False):
                result = model.forward_full_relation(**inputs)
                base = result["geometry_field"].belief_logits.float()
                logits = result["field"].belief_logits.float()
                location = cross_entropy_goal(logits, batch["target_xy"].cuda())
                preserve = F.kl_div(
                    F.log_softmax(logits, -1),
                    F.softmax(base.detach(), -1), reduction="batchmean")
                loss = location + 0.05 * preserve
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.geometry.parameters()) +
                list(model.full_relation.parameters()), 5.0)
            optimizer.step()
            size = len(batch["text"])
            seen += size
            loss_total += float(loss.detach()) * size
            distances = paired_metrics(base.detach(), logits.detach(), batch)
            corrected += int(((distances[:, 0] > 20) &
                              (distances[:, 1] <= 20)).sum())
            damaged += int(((distances[:, 0] <= 20) &
                            (distances[:, 1] > 20)).sum())
            if index % 100 == 0:
                print(json.dumps(dict(epoch=epoch+1,batch=index,seen=seen,
                                      expected=count,loss=float(loss.detach()))),
                      flush=True)
        if not a.max_batches and seen != count:
            raise RuntimeError(f"Dataset coverage incomplete: {seen}/{count}")
        row = dict(epoch=epoch+1, observed=seen, expected=count,
                   train_loss=loss_total/max(seen,1),
                   train_fixes=corrected, train_breaks=damaged,
                   wall_seconds=time.perf_counter()-started)
        history.append(row)
        ckpt = dict(
            format=FORMAT, model=model.state_dict(),
            config=original["config"], source_sha256=source_sha,
            completed_epochs=epoch+1, optimizer=optimizer.state_dict(),
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),
            python_rng=random.getstate(), numpy_rng=np.random.get_state(),
            history=history, smoke=bool(a.max_batches),
        )
        atomic_save(ckpt, a.output/"last.pt")
        atomic_json(dict(**manifest, completed_epochs=epoch+1,
                         history=history, complete=(
                            epoch+1 == a.epochs and not a.max_batches)),
                    a.output/"progress.json")
        print(json.dumps(row), flush=True)
    if a.max_batches:
        print("WARNING: Smoke checkpoint is NOT a complete full-train result",flush=True)


if __name__ == "__main__":
    main()
