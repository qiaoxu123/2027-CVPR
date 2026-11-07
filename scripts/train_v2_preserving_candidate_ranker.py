"""TRAIN-only candidate ranking head that preserves correct Geometry Top-1.

Frozen V2 Geometry supplies the exact deployment candidate logits/context.
Only the controller's candidate scoring path is trained. TRAIN GT is used only
to build a listwise loss: preserve slot 0 when it is already within 20m, use
the in-radius candidate set when slot 0 is wrong, and fall back to slot 0 when
the proposal set has no in-radius item. Path/mode/Stop weights are frozen.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from graphnav.checkpointing import atomic_json, atomic_save, file_sha256, provenance, rng_state
from graphnav.trajectory_policy import CandidateTrajectoryStop
from scripts.train_v2_candidate_controller import ControllerStates, collate_states, candidate_world


def target_distribution(candidate_world_xy, target_world):
    distance = torch.linalg.vector_norm(candidate_world_xy - target_world[:, None, :2], dim=-1)
    positive = distance <= 20.0
    labels = torch.zeros_like(distance)
    for i in range(len(distance)):
        if bool(positive[i, 0]):
            labels[i, 0] = 1.0
        elif bool(positive[i].any()):
            labels[i] = positive[i].float() / positive[i].sum()
        else:
            labels[i, 0] = 1.0
    return labels, distance


def metrics(model, loader, device, max_batches=0):
    model.eval(); records = []; start_i = 0
    with torch.no_grad():
        for b, batch in enumerate(loader):
            if max_batches and b >= max_batches:
                break
            x, y, indices, is_start = batch
            x = {k: v.to(device) for k, v in x.items()}; y = {k: v.to(device) for k, v in y.items()}
            out = model(**x)
            chosen = out.candidate_logits.float().argmax(-1)
            xy = candidate_world(x); rows = torch.arange(len(chosen), device=device)
            chosen_dist = (xy[rows, chosen] - y["target_world"][:, :2]).norm(dim=-1)
            top1_dist = (xy[:, 0] - y["target_world"][:, :2]).norm(dim=-1)
            for j in range(len(chosen)):
                if not bool(is_start[j]):
                    continue
                r20 = int(float(chosen_dist[j]) <= 20.0); top1 = int(float(top1_dist[j]) <= 20.0)
                records.append({"state_index": int(indices[j]), "slot": int(chosen[j]),
                    "NE_m": float(chosen_dist[j]), "geometry_top1_NE_m": float(top1_dist[j]),
                    "R20": r20, "geometry_top1_R20": top1})
                start_i += 1
    n = len(records)
    if not n:
        return {"n": 0}, records
    return {"n": n, "R20": sum(r["R20"] for r in records)/n,
        "NE": sum(r["NE_m"] for r in records)/n,
        "geometry_top1_R20": sum(r["geometry_top1_R20"] for r in records)/n,
        "geometry_top1_NE": sum(r["geometry_top1_NE_m"] for r in records)/n,
        "corrected": sum(r["R20"] and not r["geometry_top1_R20"] for r in records),
        "damaged": sum(r["geometry_top1_R20"] and not r["R20"] for r in records)}, records


def train_epoch(model, loader, device, optimizer, max_batches=0):
    model.train(); total = 0.; count = 0; started = time.perf_counter()
    trainable = [p for p in model.parameters() if p.requires_grad]
    torch.cuda.reset_peak_memory_stats(device)
    for b, batch in enumerate(loader):
        if max_batches and b >= max_batches:
            break
        x, y, _, _ = batch
        x = {k: v.to(device, non_blocking=True) for k, v in x.items()}
        y = {k: v.to(device, non_blocking=True) for k, v in y.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**x)
        candidate_xy = candidate_world(x)
        targets, _ = target_distribution(candidate_xy, y["target_world"])
        logits = out.candidate_logits.float()
        loss = -(targets * logits.log_softmax(-1)).sum(-1).mean()
        residual = logits - x["candidate_logits"].float()
        loss = loss + .005 * residual.square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite preserving-ranker loss")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        if b == 0 and float(grad) <= 0:
            raise RuntimeError("preserving-ranker smoke found zero gradient")
        optimizer.step(); count += len(x["pose_world"]); total += float(loss.detach()) * len(x["pose_world"])
    torch.cuda.synchronize(device); elapsed = time.perf_counter() - started
    return {"loss": total / max(count, 1), "samples": count, "seconds": elapsed,
        "samples_per_second": count / max(elapsed, 1e-9),
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("prepared-data", "data-root", "sbf-source", "stage1", "feature-cache", "controller-training", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128); p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-batches", type=int, default=0); p.add_argument("--arms", nargs="+",
        choices=("warmstart", "scratch"), default=("warmstart", "scratch"))
    a = p.parse_args(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if device.type == "cuda": torch.cuda.manual_seed_all(a.seed)
    features = torch.load(a.feature_cache / "features.pt", map_location="cpu", weights_only=True)
    private = json.loads((a.prepared_data / "labels_private.json").read_text())
    splits = {s: [r["index"] for r in private if r["split"] == s] for s in ("fit", "development")}
    if len(splits["fit"]) != 11133 or len(splits["development"]) != 3126:
        raise ValueError("TRAIN split changed")
    run = json.loads((a.controller_training / "config.json").read_text())
    if run["checkpoint_sha256"]["data_manifest"] != file_sha256(a.prepared_data / "manifest.json"):
        raise ValueError("TRAIN data differs from candidate feature experiment")
    sys.path.insert(0, str(a.sbf_source.resolve()))
    from sbfnav.dataset import CityNavDataset
    from sbfnav.mapdata import GROUND_LEVEL
    from graphnav.trajectory_policy import remaining_waypoints
    source_rows = CityNavDataset(a.data_root, "train_seen", version="processed_citynav")
    state_rows = {"fit": [], "development": []}
    for split, indices in splits.items():
        for i in indices:
            record = source_rows[i]; path = np.asarray(record.trajectory, dtype=np.float32)
            label = private[i]; ground = GROUND_LEVEL[record.map_name]
            for cut in sorted(set((0, len(path)//3, 2*len(path)//3, len(path)-1))):
                teacher = remaining_waypoints(path, cut)
                state_rows[split].append({"query_index": i, "pose_world": path[cut].tolist(),
                    "target_world": label["target_world"], "teacher_world": teacher.tolist(),
                    "valid_teacher_z": ((teacher[:, 2] >= ground+10) & (teacher[:, 2] <= ground+100)).astype(np.float32).tolist(),
                    "ground_level": ground, "is_start": cut == 0})
    datasets = {s: ControllerStates(features, rows) for s, rows in state_rows.items()}
    loader_options = dict(batch_size=a.batch_size, num_workers=0, collate_fn=collate_states,
                          pin_memory=device.type == "cuda")
    dev_loader = DataLoader(datasets["development"], shuffle=False, **loader_options)
    old = torch.load(a.stage1, map_location="cpu", weights_only=True)
    hidden = int(features["candidate_context"].shape[-1])
    old_state = {k[len("trajectory."):]: v for k, v in old["model"].items() if k.startswith("trajectory.")}
    baseline_model = CandidateTrajectoryStop(hidden=hidden).to(device).eval()
    baseline_model.load_state_dict(old_state, strict=True)
    baseline_metrics, baseline_rows = metrics(baseline_model, dev_loader, device)
    result = {"baseline_old_ranker": baseline_metrics, "baseline_rows": baseline_rows,
        "geometry_sha256": run["checkpoint_sha256"]["trained_v2"], "train_data_manifest_sha256": file_sha256(a.prepared_data / "manifest.json"),
        "dev_split": "Cambridge physical-target disjoint; 3,126 tasks; no Val/Test selection", "arms": {}}
    for arm in a.arms:
        source = torch.load(a.controller_training / arm / "last.pt", map_location="cpu", weights_only=True)
        if source.get("config", {}).get("checkpoint_sha256", {}).get("trained_v2") != result["geometry_sha256"]:
            raise ValueError("candidate controller was trained against a different Geometry checkpoint")
        torch.manual_seed(a.seed)
        model = CandidateTrajectoryStop(hidden=hidden).to(device)
        if arm == "warmstart": model.load_state_dict(old_state, strict=True)
        else: model.load_state_dict(source["model"], strict=True)
        for param in model.parameters(): param.requires_grad_(False)
        for module in (model.state, model.fusion, model.rank): module.requires_grad_(True)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=.001)
        train_loader = DataLoader(datasets["fit"], shuffle=True,
            generator=torch.Generator().manual_seed(a.seed), **loader_options)
        arm_out = a.output / arm; arm_out.mkdir(parents=True, exist_ok=True)
        best = (baseline_metrics["R20"], -baseline_metrics["NE"]); history = []
        for epoch in range(1, a.epochs + 1):
            train_metrics = train_epoch(model, train_loader, device, optimizer, a.max_batches)
            dev_metrics, dev_rows = metrics(model, dev_loader, device, a.max_batches)
            key = (dev_metrics["R20"], -dev_metrics["NE"])
            rec = {"epoch": epoch, "train": train_metrics, "dev": dev_metrics,
                "net": dev_metrics["corrected"] - dev_metrics["damaged"]}
            history.append(rec)
            if key > best:
                best = key
                atomic_save({"model": model.state_dict(), "hidden": hidden, "arm": arm, "epoch": epoch,
                    "geometry_sha256": result["geometry_sha256"], "stage1_source_sha256": file_sha256(a.stage1),
                    "selection": "TRAIN development start-state R20, tie lower NE", "provenance": provenance(),
                    "rng": rng_state()}, arm_out / "best.pt")
                atomic_json({"metrics": dev_metrics, "rows": dev_rows}, arm_out / "best_development.json")
            atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                "history": history, "best": best}, arm_out / "last.pt")
            atomic_json(history, arm_out / "history.json")
            print(json.dumps({"arm": arm, **rec}, sort_keys=True), flush=True)
            if a.max_batches: break
        result["arms"][arm] = {"best_R20": best[0], "best_NE": -best[1], "history": history}
        del model, optimizer
        if device.type == "cuda": torch.cuda.empty_cache()
    result["smoke"] = bool(a.max_batches)
    result["confirmation_evaluated"] = False
    result["val_test_evaluated"] = False
    a.output.mkdir(parents=True, exist_ok=True)
    atomic_json(result, a.output / ("smoke.json" if a.max_batches else "summary.json"))
    print(json.dumps({k: v.get("best_R20", v) for k, v in result["arms"].items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
