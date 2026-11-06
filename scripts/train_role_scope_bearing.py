"""Matched GPU fit/dev ablation: angle-only vs angle + public weak role grounding.

Fits both arms FROM SCRATCH with identical model initialization, seeds,
optimizer, epoch count and original fit/dev task split. The ONLY difference is
the GT-free intermediate-role auxiliary loss in the new arm.

Do not open PRIVATE_confirm_labels.pt, Val, Test, RGB or navigation. Existing
confirmation maps have been inspected in previous research; dev comparisons
here are exploratory and must NOT be advertised as independent confirmation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import time
from pathlib import Path

import numpy as np
import torch

from graphnav.checkpointing import atomic_json, atomic_save, file_sha256
from graphnav.relation_bearing_network import RolePolygonBearing
from graphnav.relation_scope_supervision import role_set_loss
from scripts.train_role_polygon_angles_only import angle_only
from scripts.train_role_polygon_bearing import PublicBatch, evaluate, digest


def paired_queries(left_rows, right_rows):
    """Query-equal dev comparison; common queries only."""
    def by_query(rows):
        groups = defaultdict(list)
        for r in rows:
            groups[r["index"]].append(r["top1"])
        return {k: float(np.mean(v)) for k, v in groups.items()}
    a, b = by_query(left_rows), by_query(right_rows)
    if set(a) != set(b):
        raise ValueError("Development A/B queries differ")
    delta = {k: b[k] - a[k] for k in a}
    return dict(n=len(delta), mean_delta_pp=100 * float(np.mean(list(delta.values()))),
                improved=sum(v > 0 for v in delta.values()),
                harmed=sum(v < 0 for v in delta.values()),
                unchanged=sum(v == 0 for v in delta.values()))


def fit_arm(*, arm, run, outdir, cache, weak, split, labels,
            epochs, batch_size, lr, role_weight, max_updates, resume):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = RolePolygonBearing().cuda()
    initial = digest(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
    path = outdir / arm
    if path.exists() and not resume:
        raise FileExistsError("Refuse to overwrite " + str(path))
    path.mkdir(parents=True, exist_ok=True)
    cfg = dict(format="role-scope-bearing-ablation-v1", arm=arm,
               seed=42, epochs=epochs, batch_size=batch_size, lr=lr,
               role_weight=role_weight if arm == "weak_role" else 0.,
               max_updates=max_updates, initial_digest=initial,
               public_sha256=file_sha256(run / "public_inputs.pt"),
               weak_sha256=file_sha256(outdir / "PUBLIC_weak_role_targets.pt"),
               split_sha256=file_sha256(run / "data/split.json"),
               network_sha256=file_sha256("graphnav/relation_bearing_network.py"),
               scope_sha256=file_sha256("graphnav/relation_scope_supervision.py"),
               trainer_sha256=file_sha256(__file__),
               confirm_GT_accessed=False, no_RGB_no_navigation=True)
    history, start, count = [], 0, 0
    best, best_nll = -1., float("inf")
    if resume:
        old = torch.load(path / "last.pt", weights_only=True, map_location="cpu")
        if old["config"] != cfg:
            raise ValueError("Refuse incompatible checkpoint resume")
        model.load_state_dict(old["model"], strict=True)
        opt.load_state_dict(old["optimizer"])
        history, start, count = old["history"], old["epoch"], old["updates"]
        best, best_nll = old["best"], old["best_nll"]
        torch.set_rng_state(old["rng_cpu"])
        torch.cuda.set_rng_state_all(old["rng_cuda"])
    atomic_json(cfg, path / "config.json")
    for epoch in range(start, epochs):
        order = np.random.default_rng(42 + epoch).permutation(split["fit"]).tolist()
        model.train()
        losses, angulars, roles = [], [], []
        labeled = labeled_batches = 0
        started = time.perf_counter()
        for i in range(0, len(order), batch_size):
            indices = order[i:i+batch_size]
            batch, _ = cache.build(indices)
            output = model(**batch, use_relations=True,
                           emit_steps=(arm == "weak_role"))
            angular, _ = angle_only(
                output, [labels["fit"][idx] for idx in indices], batch["anchor_valid"])
            if arm == "weak_role":
                role, matched = role_set_loss(
                    output["step_object_probabilities"],
                    [weak["targets"].get(idx, []) for idx in indices])
                if matched:
                    labeled += matched
                    labeled_batches += 1
            else:
                role = angular * 0
            loss = angular + (role_weight * role if arm == "weak_role" else 0.)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite direction/role objective")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.,
                                           error_if_nonfinite=True)
            opt.step()
            count += 1
            losses.append(float(loss.detach()))
            angulars.append(float(angular.detach()))
            roles.append(float(role.detach()))
            if count % 200 == 0:
                print(json.dumps(dict(arm=arm, epoch=epoch+1, updates=count,
                                      angular=angulars[-1], role=roles[-1])),
                      flush=True)
            if max_updates and count >= max_updates:
                break
        if not losses:
            raise RuntimeError("Training produced zero updates")
        if max_updates == 0 and arm == "weak_role" and labeled == 0:
            raise RuntimeError("No public weak supervision actually applied")
        dev_indices = split["dev"] if max_updates == 0 else split["dev"][:64]
        dev_summary, dev_rows = evaluate(
            model, cache, dev_indices, labels["dev"], "relations", batch_size)
        entry = dict(epoch=epoch+1, updates=count,
                     loss=float(np.mean(losses)), angle_loss=float(np.mean(angulars)),
                     public_role_loss=float(np.mean(roles)),
                     weak_labeled_queries=labeled,
                     weak_batches=labeled_batches,
                     train_seconds=time.perf_counter()-started,
                     dev=dev_summary)
        history.append(entry)
        improved = dev_summary["macro_Top1"] > best + 1e-9 or (
            abs(dev_summary["macro_Top1"] - best) <= 1e-9 and
            dev_summary["NLL"] < best_nll)
        if improved:
            best = dev_summary["macro_Top1"]
            best_nll = dev_summary["NLL"]
        ckpt = dict(format=cfg["format"], model=model.state_dict(),
                    optimizer=opt.state_dict(), epoch=epoch+1,
                    history=history, updates=count, config=cfg,
                    best=best, best_nll=best_nll,
                    rng_cpu=torch.get_rng_state(),
                    rng_cuda=torch.cuda.get_rng_state_all())
        atomic_save(ckpt, path / "last.pt")
        if improved:
            atomic_save(ckpt, path / "best.pt")
            with (path / "best_dev_rows.jsonl").open("w") as f:
                for row in dev_rows:
                    f.write(json.dumps(row) + "\n")
        atomic_json(history, path / "history.json")
        print(json.dumps(dict(arm=arm, **entry)), flush=True)
        if max_updates and count >= max_updates:
            break
    return dict(arm=arm, initial_digest=initial, history=history)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True,
                        help="Existing relation-bearings run with prepared public features")
    parser.add_argument("--output", type=Path, required=True,
                        help="Directory containing PUBLIC_weak_role_targets.pt")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--role-weight", type=float, default=.2)
    parser.add_argument("--max-updates", type=int, default=0,
                        help="Smoke only; not a publishable comparison")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not 0 < args.lr < 1 or (
            not 0 < args.role_weight <= 1) or args.max_updates < 0:
        raise ValueError("Invalid fixed training arguments")
    if not torch.cuda.is_available():
        raise RuntimeError("Requires the CityNav GPU host")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    splits = json.loads((args.run / "data/split.json").read_text())
    weak = torch.load(args.output / "PUBLIC_weak_role_targets.pt",
                      weights_only=True, map_location="cpu")
    if weak["format"] != "public-role-weak-targets-v1" or (
            weak["fit"] != splits["indices"]["fit"] or
            weak["dev"] != splits["indices"]["dev"]):
        raise ValueError("Public weak targets do not match TRAIN fit/dev indices")
    # Deliberately DO NOT load PRIVATE_confirm_labels.pt.
    labels = {name: torch.load(args.run / ("PRIVATE_" + name + "_labels.pt"),
                               weights_only=True, map_location="cpu")
              for name in ("fit", "dev")}
    cache = PublicBatch(args.run)
    results = {}
    for arm in ("angle_only", "weak_role"):
        results[arm] = fit_arm(
            arm=arm, run=args.run, outdir=args.output, cache=cache,
            weak=weak, split=splits["indices"], labels=labels,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, role_weight=args.role_weight,
            max_updates=args.max_updates, resume=args.resume)
    if results["angle_only"]["initial_digest"] != results["weak_role"]["initial_digest"]:
        raise AssertionError("A/B parameters were not identically initialized")
    if args.max_updates:
        print("SMOKE ONLY: dev metrics are not complete.", flush=True)
        return
    a = [json.loads(line) for line in
         (args.output/"angle_only"/"best_dev_rows.jsonl").read_text().splitlines()]
    b = [json.loads(line) for line in
         (args.output/"weak_role"/"best_dev_rows.jsonl").read_text().splitlines()]
    summary = dict(
        format="role-scope-bearing-ablation-v1",
        public_role_coverage=json.loads(
            (args.output/"PUBLIC_role_coverage.json").read_text()),
        controls={
            name: dict(best_dev=max(r["history"], key=lambda h:
                                (h["dev"]["macro_Top1"], -h["dev"]["NLL"]))["dev"],
                       initial_digest=r["initial_digest"])
            for name, r in results.items()},
        paired_dev=paired_queries(a, b),
        no_confirmation_GT_read=True, old_confirm_maps_not_reused_for_validation=True,
        dev_is_for_model_selection_not_independent_confirmation=True,
        weak_scope_labels_are_not_human_truth=True,
        no_RGB_no_navigation=True)
    atomic_json(summary, args.output/"DEVELOPMENT_REPORT.json")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
