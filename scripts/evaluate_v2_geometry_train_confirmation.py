"""Compare frozen V2 Geometry checkpoints on the unused Birmingham TRAIN group.
This is an adaptation-holdout diagnostic; the source V2 was pretrained on all TRAIN.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from graphnav.checkpointing import atomic_json, file_sha256
from graphnav.candidate_geometry import collate_full
from graphnav.full_experiment import _batch_inputs
from graphnav.token_language import FrozenBERTTokens
from scripts.train_v2_top1_geometry import Indexed
from scripts.unified_pgvb_v3_pilot import _load_base


def geometry_state(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = state.get("model", {})
    weights = {k[len("geometry."):]: v for k, v in model.items() if k.startswith("geometry.")}
    if not weights and "geometry" in state:
        weights = state["geometry"]
    if not weights:
        raise ValueError(f"No Geometry weights found in {path}")
    return state, weights


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "source-v2", "bert-cache", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    p.add_argument("--split", choices=("development", "confirmation"), default="confirmation")
    p.add_argument("--batch-size", type=int, default=32)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this evaluation")
    device = "cuda"
    labels = json.loads((a.prepared / "labels_private.json").read_text())
    indices = [r["index"] for r in labels if r["split"] == a.split]
    expected = {"development": 3126, "confirmation": 7619}[a.split]
    if len(indices) != expected:
        raise ValueError(f"Expected {expected} {a.split} items, found {len(indices)}")
    split_groups = {name: {r["group"] for r in labels if r["split"] == name}
                    for name in ("fit", "development", "confirmation")}
    if any(split_groups[x] & split_groups[y]
           for x, y in (("fit", "development"), ("fit", "confirmation"),
                        ("development", "confirmation"))):
        raise ValueError("Physical-target group leakage across TRAIN splits")

    model, cfg = _load_base(a.source_v2, device)
    ds = Indexed(a.prepared / "public_train.jsonl", grid_size=28,
        max_landmarks=cfg.max_landmarks, max_roads=cfg.max_roads,
        max_relations=cfg.max_relations, prior_sigma_m=cfg.prior_sigma_m,
        rule_sigma_m=cfg.rule_sigma_m, deduplicate_queries=False)
    bert = FrozenBERTTokens(cfg.text_backbone, device=device,
        max_length=getattr(cfg, "max_text_length", 96))
    bert.load_cache(a.bert_cache)
    loader = DataLoader(ds, batch_size=a.batch_size, sampler=indices,
        num_workers=0, collate_fn=collate_full, pin_memory=True)
    a.output.mkdir(parents=True, exist_ok=True)
    all_summaries = {}
    for checkpoint in a.checkpoints:
        state, weights = geometry_state(checkpoint)
        if state.get("config", {}).get("geometry_args") not in (None, vars(cfg)):
            raise ValueError(f"Geometry configuration mismatch: {checkpoint}")
        if "bert_model" in state:
            raise ValueError("Joint BERT checkpoint requires its matching encoder; refusing cached BERT features")
        model.load_state_dict(weights, strict=True)
        model.eval().requires_grad_(False)
        rows = []
        started = time.perf_counter()
        with torch.no_grad():
            for batch in loader:
                obs, tokens, mask, nodes = _batch_inputs(batch, bert, device, "geometry")
                logits = model(obs, variant="geometry", tokens=tokens,
                    token_mask=mask, node_emb=nodes).belief_logits.float()
                row_ids = batch["row_index"].tolist()
                bounds = torch.tensor([ds.records[i]["world_bounds"] for i in row_ids], device=device)
                target = torch.tensor([labels[i]["target_world"] for i in row_ids], device=device)
                xy = obs["grid_xy"][torch.arange(len(logits), device=device), logits.argmax(-1)]
                world = bounds[:, :2] + xy * (bounds[:, 2:] - bounds[:, :2])
                errors = (world - target[:, :2]).norm(dim=-1).cpu().tolist()
                rows.extend({"index": i, "NE_m": e, "R20": int(e <= 20)} for i, e in zip(row_ids, errors))
        summary = {"checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
            "n": len(rows), "R20": float(np.mean([r["R20"] for r in rows])),
            "NE_m": float(np.mean([r["NE_m"] for r in rows])),
            "elapsed_s": time.perf_counter() - started}
        key = checkpoint.stem
        all_summaries[key] = summary
        atomic_json(rows, a.output / f"{key}_rows.json")
        print(json.dumps(summary), flush=True)
    atomic_json({"split": a.split, "count": expected,
        "physical_target_groups": len(split_groups[a.split]), "seed": 42,
        "precision": "FP32; TF32 disabled", "GT": "TRAIN scoring only, after forward",
        "public_inputs": "public_train.jsonl with neutral target placeholder",
        "caveat": "Base V2 was pretrained on all official TRAIN, including this group; this only measures adaptation transfer/regression, not unseen-pretraining generalization.",
        "prepared_manifest_sha256": file_sha256(a.prepared / "manifest.json"),
        "parser_sha256": file_sha256(Path("graphnav/public_landmark_mentions.py")),
        "summaries": all_summaries}, a.output / "summary.json")


if __name__ == "__main__":
    main()
