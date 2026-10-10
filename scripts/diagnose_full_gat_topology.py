"""Inference-only topology sensitivity, using identical frozen weights/inputs.

This is not an independently retrained ablation and must not select a model.
"""

import argparse
import json
from pathlib import Path

import torch

from graphnav.full_experiment import evaluate, make_loader, new_model, static_signature
from graphnav.token_language import FrozenBERTTokens


class SelfOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch, **kwargs):
        batch = dict(batch)
        size = batch["adjacency"].shape[-1]
        batch["adjacency"] = batch["adjacency"] & torch.eye(size, device=batch["adjacency"].device, dtype=torch.bool)[None]
        return self.model(batch, **kwargs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output", required=True)
    cli = p.parse_args()
    state = torch.load(cli.checkpoint, map_location="cpu", weights_only=True)
    args = argparse.Namespace(**state["args"])
    args.device = "cuda"
    args.batch_size = 16
    args.workers = 4
    device = torch.device(args.device)
    model = new_model(args, device).eval()
    model.load_state_dict(state["model"])
    encoder = FrozenBERTTokens(name=args.text_backbone, device=args.device,
                               max_length=getattr(args, "max_text_length", 96))
    output = Path(cli.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {"interpretation": "Post-hoc inference sensitivity; no retraining or model selection",
              "checkpoint": cli.checkpoint, "metrics": {}}
    for split in ("val_seen", "val_unseen"):
        loader = make_loader(Path(cli.data_dir) / f"graphnav_{split}.jsonl", args)
        assert static_signature(loader.dataset) == state["static_signatures"][split]
        report["metrics"][split] = {}
        for name, arm in (("full", model), ("self_only", SelfOnly(model))):
            report["metrics"][split][name] = evaluate(arm, loader, encoder, None, None,
                args, "gat", device, output / f"{name}_{split}_queries.jsonl")
        print(json.dumps({split: report["metrics"][split]}), flush=True)
    (output / "topology_metrics.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
