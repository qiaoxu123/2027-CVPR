"""Optional second stage: current-frame PGVB fine-tuning in full relation model.

Stage 1 already trains Geometry + learned relation on ALL revised TRAIN queries.
This stage trains the existing PGVB on actual TRAIN observed RGB tensors from
the verified direct-target cache; it does NOT change the full-task denominator
or claim the 573-query RGB cache covers all CityNav instructions.

The geometry and relation layers are held fixed here to preserve the complete
TRAIN pretraining. No future view or target coordinate enters model.forward.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from graphnav.checkpointing import atomic_json, atomic_save, file_sha256
from graphnav.direct_target_data import batch_inputs
from graphnav.full_relation_training import FORMAT, load_full_relation_model


def visual_rows(inventory, cache):
    tasks = json.loads((inventory/"public_tasks.json").read_text())
    labels = json.loads((inventory/"private_labels.json").read_text())
    kept, ledger = [], []
    for task in tasks:
        if task["subset"] != "fit":
            continue
        anchors = task.get("anchors", [])
        # Current PGVB is a one-landmark visual scorer. Multi-reference
        # semantics stay active in Geometry/Relation but do not impersonate
        # a visually verified target identity in this RGB training phase.
        if len(anchors) != 1 or len(anchors[0]["landmark"].get("public_ids", [])) != 1:
            ledger.append(dict(index=task["index"], reason="ambiguous_visual_anchor"))
            continue
        item = torch.load(cache/f"{task['index']:05d}.pt",
                          map_location="cpu", weights_only=True)
        if item["index"] != task["index"] or item["subset"] != task["subset"]:
            raise ValueError("Mismatched TRAIN visual cache")
        for obs in item["observations"]:
            if obs.get("observed") is not None:
                patch_labels = obs.get('anchor_labels')
                valid = obs['observed']['valid'].bool()
                if patch_labels is None or int(((patch_labels==1)&valid).sum())<2:
                    ledger.append(dict(index=task['index'],reason='no_runtime_supported_anchor_ROI'))
                    continue
                kept.append(dict(query=item["query"], task=task,
                                 observed=obs["observed"],
                                 label=labels[task["index"]]["target_world"]))
            else:
                ledger.append(dict(index=task["index"], reason="no_causal_rgb"))
    if not kept:
        raise ValueError("No validated causal TRAIN RGB observations available")
    return kept, ledger


def target_norm(rows, device):
    bounds = torch.as_tensor([r["task"]["world_bounds"] for r in rows],
                             device=device, dtype=torch.float32)
    targets = torch.as_tensor([r["label"] for r in rows],
                              device=device, dtype=torch.float32)
    return (targets - bounds[:, :2]) / (bounds[:, 2:] - bounds[:, :2])


def main():
    p = argparse.ArgumentParser()
    for name in ("base-checkpoint", "inventory", "cache", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    a=p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA needed for cached RGB PGVB stage")
    if a.epochs < 1 or a.batch_size < 1:
        raise ValueError("Invalid visual training configuration")
    torch.set_num_threads(4)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    a.output.mkdir(parents=True, exist_ok=True)

    source = torch.load(a.base_checkpoint, map_location="cpu", weights_only=True)
    if source.get("format") != FORMAT or source.get("completed_epochs", 0) < 1 or source.get("smoke"):
        raise ValueError("Require full completed TRAIN Geometry+Relation checkpoint")
    rows, rejected = visual_rows(a.inventory, a.cache)
    fingerprint = dict(
        source_checkpoint_sha256=file_sha256(a.base_checkpoint),
        public_inventory_sha256=file_sha256(a.inventory/"public_tasks.json"),
        private_train_label_sha256=file_sha256(a.inventory/"private_labels.json"),
        visual_cache_manifest_sha256=file_sha256(a.cache/"manifest.json"),
        visual_rows=len(rows), ineligible_visual_rows=len(rejected),
        geometry_relation_frozen=True, PGVB_trainable=True, seed=a.seed,
        epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
        train_only=True, Val_Test_used=False,
    )
    manifest = a.output/"manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != fingerprint:
        raise ValueError("RGB checkpoint provenance changed")
    if a.resume and not manifest.exists():
        raise ValueError("Resume requires existing manifest")
    if not a.resume and (a.output/"last.pt").exists():
        raise FileExistsError("Existing visual checkpoint; specify --resume")
    atomic_json(fingerprint,manifest)
    atomic_json(rejected,a.output/"ineligible_visual_ledger.json")

    previous = torch.load(a.output/"last.pt",map_location="cpu",
                          weights_only=True) if a.resume else None
    state = previous if previous is not None else source
    model = load_full_relation_model(state,"cuda")
    model.requires_grad_(False)
    model.pgvb.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.pgvb.parameters(),lr=a.lr,weight_decay=0.001)
    if previous:
        optimizer.load_state_dict(previous["visual_optimizer"])
        torch.set_rng_state(previous["torch_rng"])
        torch.cuda.set_rng_state_all(previous["cuda_rng"])
        random.setstate(previous["python_rng"])
    start_epoch = int(previous.get("visual_epochs",0)) if previous else 0
    history = list(previous.get("visual_history",[])) if previous else []
    start=time.perf_counter()
    original_frozen={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith('pgvb.')}
    for epoch in range(start_epoch,a.epochs):
        order=np.random.default_rng(a.seed+epoch).permutation(len(rows)).tolist()
        # Keep the top-level model in train mode to forbid the static
        # cross-query cache, while frozen core dropout is disabled.
        model.train()
        model.geometry.eval()
        model.full_relation.eval()
        model.pgvb.train()
        total = used = optimized = inactive = 0
        for off in range(0,len(rows),a.batch_size):
            group = [rows[i] for i in order[off:off+a.batch_size]]
            inp = batch_inputs(group,device="cuda",visual=True)
            inputs = dict(
                static_inputs=inp["static_inputs"], tokens=inp["tokens"],
                token_mask=inp["token_mask"], node_emb=inp["node_emb"],
            )
            with torch.autocast("cuda",enabled=False):
                visual_result = model.forward_full_relation(
                    **inputs,observed=inp["observed"],
                )
                visual = visual_result["field"].belief_logits.float()
                with torch.no_grad():
                    static = model.forward_full_relation(**inputs)["field"].belief_logits.float()
                targets = target_norm(group,"cuda")
                from graphnav.static_localization import cross_entropy_goal
                loss = cross_entropy_goal(visual,targets) + 0.1 * F.kl_div(
                    F.log_softmax(visual,-1),F.softmax(static,-1),
                    reduction="batchmean")
            used += len(group)
            if not visual_result["audit"]["active"] or not loss.requires_grad:
                inactive += len(group)
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite PGVB loss')
            grad=sum(float(p.grad.abs().sum()) for p in model.pgvb.parameters() if p.grad is not None)
            if not np.isfinite(grad):raise FloatingPointError('Nonfinite PGVB gradients')
            torch.nn.utils.clip_grad_norm_(model.pgvb.parameters(),5.,error_if_nonfinite=True)
            optimizer.step()
            optimized += len(group)
            total += float(loss.detach())*len(group)
        if used != len(rows):
            raise RuntimeError("RGB cache sample coverage incomplete")
        if not optimized:
            raise RuntimeError("No active differentiable RGB evidence; stop visual fine-tuning")
        assert all(torch.equal(model.state_dict()[k].cpu(),v) for k,v in original_frozen.items())
        report=dict(epoch=epoch+1,train_rgb_frames=used,
                    optimized_rgb_frames=optimized, inactive_rgb_frames=inactive,
                    loss=total/optimized,seconds=time.perf_counter()-start,
                    frozen_geometry_relation_unchanged=True,last_gradient_L1=grad)
        history.append(report)
        output=dict(
            format=FORMAT,model=model.state_dict(),
            config=source["config"],source_sha256=source["source_sha256"],
            completed_epochs=source["completed_epochs"],
            planned_epochs=source["planned_epochs"],
            visual_epochs=epoch+1,visual_history=history,
            visual_optimizer=optimizer.state_dict(),
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),
            python_rng=random.getstate(),
            smoke=False,
        )
        atomic_save(output,a.output/"last.pt")
        atomic_json(dict(**fingerprint,completed_visual_epochs=epoch+1,
                         complete=epoch+1 == a.epochs,history=history),
                    a.output/"progress.json")
        print(json.dumps(report),flush=True)


if __name__ == "__main__":
    main()
