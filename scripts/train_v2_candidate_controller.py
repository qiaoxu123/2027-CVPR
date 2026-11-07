"""Train a candidate-aligned controller on frozen V2 Geometry features.

Geometry/BERT and perception checkpoints stay frozen. All model-facing
features are generated from public inputs; TRAIN target and teacher paths are
kept in separate loss labels. This adapts the candidate head to the V2 proposal
distribution instead of assuming the older Stage1 candidate distribution is
identical.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from graphnav.candidate_geometry import FullCandidateDataset, collate_full
from graphnav.checkpointing import atomic_json, atomic_save, file_sha256, provenance, rng_state
from graphnav.full_experiment import _batch_inputs
from graphnav.trajectory_policy import CandidateTrajectoryStop, remaining_waypoints, trajectory_losses
from graphnav.token_language import FrozenBERTTokens
from scripts.unified_pgvb_v3_pilot import _load_base


class IndexedFullCandidateDataset(FullCandidateDataset):
    def __getitem__(self, index):
        row = super().__getitem__(index)
        row["row_index"] = index
        return row


class ControllerStates(Dataset):
    def __init__(self, features, states):
        self.features = features
        self.states = states

    def __len__(self):
        return len(self.states)

    def __getitem__(self, i):
        s = self.states[i]
        q = s["query_index"]
        x = {key: self.features[key][q].float() for key in
             ("candidate_xy", "candidate_context", "candidate_logits", "bounds", "belief_entropy")}
        x.update(pose_world=torch.tensor(s["pose_world"], dtype=torch.float32),
                 ground_level=torch.tensor(s["ground_level"], dtype=torch.float32),
                 observed_fraction=torch.tensor(0., dtype=torch.float32))
        y = {key: torch.tensor(s[key], dtype=torch.float32) for key in
             ("target_world", "teacher_world", "valid_teacher_z")}
        return x, y, torch.tensor(i), torch.tensor(s["is_start"])


def collate_states(rows):
    inputs = {k: torch.stack([r[0][k] for r in rows]) for k in rows[0][0]}
    labels = {k: torch.stack([r[1][k] for r in rows]) for k in rows[0][1]}
    return inputs, labels, torch.stack([r[2] for r in rows]), torch.stack([r[3] for r in rows])


def move(batch, device):
    x, y, idx, is_start = batch
    return ({k: v.to(device, non_blocking=True) for k, v in x.items()},
            {k: v.to(device, non_blocking=True) for k, v in y.items()},
            idx.to(device), is_start.to(device))


def candidate_world(inputs):
    return inputs["bounds"][:, None, :2] + inputs["candidate_xy"] * (
        inputs["bounds"][:, None, 2:] - inputs["bounds"][:, None, :2])


def train_epoch(model, loader, device, optimizer, max_batches=0):
    model.train(); sums = {}; n = 0; start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for b, batch in enumerate(loader):
        if max_batches and b >= max_batches:
            break
        x, y, _, _ = move(batch, device)
        optimizer.zero_grad(set_to_none=True)
        out = model(**x)
        out.paths_world = out.paths_world.float(); out.mode_logits = out.mode_logits.float()
        out.candidate_logits = out.candidate_logits.float(); out.stop_logits = out.stop_logits.float()
        loss = trajectory_losses(out, candidate_world_xy=candidate_world(x), pose_world=x["pose_world"],
                                 **y, stop_pos_weight=torch.tensor(20., device=device))
        if not torch.isfinite(loss["total"]):
            raise FloatingPointError("nonfinite candidate-controller loss")
        loss["total"].backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        if b == 0 and float(grad) <= 0:
            raise RuntimeError("candidate-controller gradient is zero")
        optimizer.step(); size = len(x["pose_world"]); n += size
        for k, v in loss.items():
            sums[k] = sums.get(k, 0.) + float(v.detach()) * size
    torch.cuda.synchronize(device); seconds = time.perf_counter() - start
    return {**{k: v / max(n, 1) for k, v in sums.items()}, "samples": n, "seconds": seconds,
            "samples_per_second": n / max(seconds, 1e-9),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30}


@torch.no_grad()
def evaluate(model, loader, device, *, max_batches=0):
    model.eval(); errs = []; joint_errs = []; base_errs = []; records = []; starts = 0
    for b, batch in enumerate(loader):
        if max_batches and b >= max_batches:
            break
        x, y, indices, is_start = move(batch, device)
        out = model(**x); out.candidate_logits = out.candidate_logits.float(); out.mode_logits = out.mode_logits.float()
        joint_selected, _, _, _ = out.select()
        selected = out.candidate_logits.argmax(-1)
        world = candidate_world(x); row = torch.arange(len(selected), device=device)
        selected_error = (world[row, selected] - y["target_world"][:, :2]).norm(dim=-1)
        joint_error = (world[row, joint_selected] - y["target_world"][:, :2]).norm(dim=-1)
        top1_error = (world[:, 0] - y["target_world"][:, :2]).norm(dim=-1)
        for j in range(len(selected)):
            if not bool(is_start[j]):
                continue
            i = int(indices[j]); e = float(selected_error[j]); je = float(joint_error[j]); b0 = float(top1_error[j]); starts += 1
            errs.append(e); joint_errs.append(je); base_errs.append(b0)
            records.append({"state_index": i, "selected_slot": int(selected[j]), "selected_NE_m": e,
                            "joint_selected_slot": int(joint_selected[j]), "joint_selected_NE_m": je,
                            "geometry_top1_NE_m": b0, "selected_R20": int(e <= 20),
                            "joint_selected_R20": int(je <= 20),
                            "geometry_top1_R20": int(b0 <= 20)})
    if not errs:
        return {"n": 0, "R20": 0., "NE": 0.}, records
    return {"n": starts, "R20": float(np.mean(np.asarray(errs) <= 20)), "NE": float(np.mean(errs)),
            "joint_R20": float(np.mean(np.asarray(joint_errs) <= 20)), "joint_NE": float(np.mean(joint_errs)),
            "geometry_top1_R20": float(np.mean(np.asarray(base_errs) <= 20)),
            "geometry_top1_NE": float(np.mean(base_errs)),
            "corrected": int(sum(r["selected_R20"] and not r["geometry_top1_R20"] for r in records)),
            "damaged": int(sum(r["geometry_top1_R20"] and not r["selected_R20"] for r in records))}, records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("prepared-data", "data-root", "sbf-source", "source-v2", "trained-v2", "bert-cache",
                "stage1", "feature-cache", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64); p.add_argument("--feature-batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4); p.add_argument("--max-batches", type=int, default=0,
                   help="Smoke only: limit train and validation batches")
    p.add_argument("--arms", nargs="+", choices=("warmstart", "scratch"), default=("warmstart", "scratch"))
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda"); out = a.output; out.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False; torch.set_float32_matmul_precision("highest")

    manifest = json.loads((a.prepared_data / "manifest.json").read_text())
    private = json.loads((a.prepared_data / "labels_private.json").read_text())
    public_path = a.prepared_data / "public_train.jsonl"
    if manifest.get("n") != 21878 or len(private) != 21878:
        raise ValueError("Expected complete Revised TRAIN records")
    public = [json.loads(line) for line in public_path.read_text().splitlines()]
    if len(public) != 21878 or any(r.get("target_xy") != [0., 0.] for r in public):
        raise ValueError("Public queries must be complete and use neutral target placeholders")
    splits = {s: [r["index"] for r in private if r["split"] == s]
              for s in ("fit", "development", "confirmation")}
    fit_groups = {private[i]["group"] for i in splits["fit"]}
    dev_groups = {private[i]["group"] for i in splits["development"]}
    if fit_groups & dev_groups or len(splits["fit"]) != 11133 or len(splits["development"]) != 3126:
        raise ValueError("Physical-target split isolation/counts changed")

    sys.path.insert(0, str(a.sbf_source.resolve()))
    from sbfnav.dataset import CityNavDataset
    from sbfnav.mapdata import GROUND_LEVEL
    from graphnav.trajectory_policy import remaining_waypoints
    source_rows = CityNavDataset(a.data_root, "train_seen", version="processed_citynav")
    if len(source_rows) != 21878:
        raise ValueError("TRAIN annotation count mismatch")
    for i in range(len(private)):
        source = source_rows[i]
        if source.map_name != private[i]["episode"][0] or source.map_name != public[i]["map_name"]:
            raise ValueError(f"episode/map alignment mismatch at {i}")
        if source.instruction != public[i]["instruction"]:
            raise ValueError(f"instruction alignment mismatch at {i}")

    geometry, cfg = _load_base(a.source_v2, device)
    trained = torch.load(a.trained_v2, map_location="cpu", weights_only=True)
    geom_state = {k[len("geometry."):]: v for k, v in trained["model"].items() if k.startswith("geometry.")}
    geometry.load_state_dict(geom_state, strict=True); geometry.eval().requires_grad_(False)
    bert = FrozenBERTTokens(cfg.text_backbone, device=device, max_length=getattr(cfg, "max_text_length", 96))
    bert.load_cache(a.bert_cache)
    queries = IndexedFullCandidateDataset(public_path, grid_size=cfg.grid_size, max_landmarks=cfg.max_landmarks,
        max_roads=cfg.max_roads, max_relations=cfg.max_relations, prior_sigma_m=cfg.prior_sigma_m,
        rule_sigma_m=cfg.rule_sigma_m, deduplicate_queries=False)
    if len(queries) != 21878:
        raise ValueError("V2 query dataset count mismatch")

    feature_path = a.feature_cache / "features.pt"; feature_manifest_path = a.feature_cache / "manifest.json"
    feature_meta = {"n": len(queries), "source_v2_sha256": file_sha256(a.source_v2),
        "trained_v2_sha256": file_sha256(a.trained_v2), "public_sha256": file_sha256(public_path),
        "bert_cache_sha256": file_sha256(a.bert_cache), "precision": "same BF16 autocast geometry path as runtime; cached logits FP32/context FP16",
        "label_in_feature_generation": False, "script_sha256": file_sha256(Path(__file__))}
    a.feature_cache.mkdir(parents=True, exist_ok=True)
    feature_inputs = {k: v for k, v in feature_meta.items() if k != "script_sha256"}
    if feature_manifest_path.exists():
        saved = json.loads(feature_manifest_path.read_text())
        saved_inputs = {k: v for k, v in saved.items() if k not in ("complete", "script_sha256")}
        if saved_inputs != feature_inputs:
            raise ValueError("Feature-cache source changed; refusing to reuse directory")
    if feature_path.exists():
        fm = json.loads(feature_manifest_path.read_text())
        if not fm.get("complete"):
            raise ValueError("Feature cache fingerprint mismatch")
        features = torch.load(feature_path, map_location="cpu", weights_only=True)
    else:
        atomic_json(dict(feature_meta, complete=False), feature_manifest_path)
        collected = {k: [] for k in ("candidate_xy", "candidate_context", "candidate_logits", "bounds", "belief_entropy")}
        loader = DataLoader(queries, batch_size=a.feature_batch_size, shuffle=False, num_workers=0,
                            collate_fn=collate_full, pin_memory=True)
        with torch.no_grad():
            for bi, batch in enumerate(loader):
                x, tokens, token_mask, names = _batch_inputs(batch, bert, device, "geometry")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    field = geometry(x, variant="geometry", tokens=tokens, token_mask=token_mask, node_emb=names)
                logits = field.belief_logits.float(); ids = field.candidate_ids
                probs = logits.softmax(-1)
                entropy = -(probs * probs.clamp_min(1e-8).log()).sum(-1) / math.log(logits.shape[-1])
                rows = batch["row_index"].tolist()
                bounds = torch.tensor([queries.records[i]["world_bounds"] for i in rows], dtype=torch.float32)
                vals = {"candidate_xy": field.candidate_xy.float().cpu(),
                    "candidate_context": field.candidate_context.half().cpu(),
                    "candidate_logits": logits.gather(1, ids).cpu(), "bounds": bounds,
                    "belief_entropy": entropy.cpu()}
                for key, value in vals.items():
                    collected[key].append(value)
                if bi == 0 or (bi + 1) % 100 == 0:
                    print(json.dumps({"feature_batches": bi + 1, "queries": min((bi + 1) * a.feature_batch_size, len(queries))}), flush=True)
        features = {k: torch.cat(v, 0) for k, v in collected.items()}
        atomic_save(features, feature_path); atomic_json(dict(feature_meta, complete=True), feature_manifest_path)

    # Four causal states per human route; no future frame is an inference input.
    states = {"fit": [], "development": []}
    for split in states:
        for index in splits[split]:
            r = source_rows[index]; path = np.asarray(r.trajectory, dtype=np.float32)
            if len(path) == 0:
                continue
            cuts = sorted(set((0, len(path)//3, 2*len(path)//3, len(path)-1)))
            label = private[index]
            for cut in cuts:
                teacher = remaining_waypoints(path, cut)
                ground = GROUND_LEVEL[r.map_name]
                states[split].append({"query_index": index, "pose_world": path[cut].tolist(),
                    "target_world": label["target_world"], "teacher_world": teacher.tolist(),
                    "valid_teacher_z": ((teacher[:, 2] >= ground + 10) & (teacher[:, 2] <= ground + 100)).astype(np.float32).tolist(),
                    "ground_level": ground, "is_start": cut == 0})
    feature_sets = {s: ControllerStates(features, rows) for s, rows in states.items()}
    config = {"seed": a.seed, "epochs": a.epochs, "batch_size": a.batch_size, "lr": a.lr,
        "arms": a.arms, "fit_episodes": len(splits["fit"]), "dev_episodes": len(splits["development"]),
        "fit_states": len(states["fit"]), "dev_states": len(states["development"]),
        "split_unit": "physical target group; fit/dev disjoint", "geometry_frozen": True, "bert_frozen": True,
        "RGB": False, "GT": "TRAIN labels only for trajectory/ranking losses and post-hoc dev scoring",
        "loss": "existing trajectory_losses: candidate target-distance ranking + expert future path/mode + Stop",
        "checkpoint_sha256": {"trained_v2": file_sha256(a.trained_v2), "stage1": file_sha256(a.stage1),
            "data_manifest": file_sha256(a.prepared_data / "manifest.json"), "feature_cache": file_sha256(feature_manifest_path)},
        "dev_selection": "start-state candidate R20, tie lower NE; no Val/Test during selection",
        "caveat": manifest.get("prior_checkpoint_caveat", "")}
    atomic_json(config, out / "config.json")
    loader_options = dict(batch_size=a.batch_size, num_workers=0, collate_fn=collate_states, pin_memory=True)
    dev_loader = DataLoader(feature_sets["development"], shuffle=False, **loader_options)
    old = torch.load(a.stage1, map_location="cpu", weights_only=True)
    old_state = {k[len("trajectory."):]: v for k, v in old["model"].items() if k.startswith("trajectory.")}
    hidden = int(features["candidate_context"].shape[-1])
    old_hidden = old.get("hidden", old.get("a3_args", {}).get("hidden"))
    if old_hidden != hidden:
        raise ValueError(f"Stage1 head hidden={old_hidden} differs from V2 context={hidden}")

    baseline = CandidateTrajectoryStop(hidden=hidden).to(device)
    baseline.load_state_dict(old_state, strict=True)
    base_dev, base_rows = evaluate(baseline, dev_loader, device, max_batches=a.max_batches)
    atomic_json({"metrics": base_dev, "rows": base_rows}, out / "old_ranker_development.json")
    summary = {"old_ranker": base_dev}; started_all = time.perf_counter()
    for arm in a.arms:
        torch.manual_seed(a.seed)
        model = CandidateTrajectoryStop(hidden=hidden).to(device)
        if arm == "warmstart":
            model.load_state_dict(old_state, strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.001)
        train_loader = DataLoader(feature_sets["fit"], shuffle=True,
                                  generator=torch.Generator().manual_seed(a.seed), **loader_options)
        arm_out = out / arm; arm_out.mkdir(exist_ok=True)
        best_key = (base_dev.get("R20", 0.), -base_dev.get("NE", float("inf")))
        history = []; started = time.perf_counter()
        for epoch in range(1, a.epochs + 1):
            train_metrics = train_epoch(model, train_loader, device, optimizer, max_batches=a.max_batches)
            dev_metrics, dev_rows = evaluate(model, dev_loader, device, max_batches=a.max_batches)
            key = (dev_metrics.get("R20", 0.), -dev_metrics.get("NE", float("inf")))
            record = {"epoch": epoch, "train": train_metrics, "dev": dev_metrics,
                "corrected": dev_metrics.get("corrected", 0), "damaged": dev_metrics.get("damaged", 0),
                "net": dev_metrics.get("corrected", 0) - dev_metrics.get("damaged", 0),
                "elapsed_s": time.perf_counter() - started}
            history.append(record); improved = key > best_key
            if improved:
                best_key = key
                atomic_save({"model": model.state_dict(), "hidden": hidden, "arm": arm, "epoch": epoch,
                    "geometry_sha256": file_sha256(a.trained_v2), "stage1_source_sha256": file_sha256(a.stage1),
                    "config": config, "provenance": provenance(), "rng": rng_state()}, arm_out / "best.pt")
                atomic_json({"metrics": dev_metrics, "rows": dev_rows}, arm_out / "best_development.json")
            atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                "history": history, "best_key": best_key, "config": config, "rng": rng_state()}, arm_out / "last.pt")
            atomic_json(history, arm_out / "history.json")
            print(json.dumps({"arm": arm, **record}, sort_keys=True), flush=True)
            if a.max_batches:
                break
        summary[arm] = {"best_R20": best_key[0], "best_NE": -best_key[1], "history": history}
        del model, optimizer; torch.cuda.empty_cache()
    result = {"complete": not bool(a.max_batches), "smoke": bool(a.max_batches), "summary": summary,
        "elapsed_s": time.perf_counter() - started_all, "confirmation_evaluated": False,
        "val_test_evaluated": False}
    atomic_json(result, out / ("smoke.json" if a.max_batches else "summary.json"))


if __name__ == "__main__":
    main()
