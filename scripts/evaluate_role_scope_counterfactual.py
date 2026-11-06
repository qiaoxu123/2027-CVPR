"""Development-only frozen diagnostic: scramble only public intermediate-role relation phrases.

No training, no GT in feature construction, no historical confirmation/Val/Test.
A drop in angle accuracy is evidence of reliance on intermediate relation text,
not proof that the parser's scope is semantically correct.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from graphnav.checkpointing import atomic_json
from graphnav.relation_bearing_network import RolePolygonBearing
from scripts.role_bearing_evaluation import evaluate_cpu_metrics
from scripts.train_role_polygon_bearing import PublicBatch
from scripts.train_role_scope_bearing import paired_queries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    a = parser.parse_args()
    torch.set_num_threads(4)
    split = json.loads((a.run/"data/split.json").read_text())
    dev_ids = split["indices"]["dev"]
    dev_labels = torch.load(a.run/"PRIVATE_dev_labels.pt",
                            weights_only=True, map_location="cpu")
    weak = torch.load(a.output/"PUBLIC_weak_role_targets.pt",
                      weights_only=True, map_location="cpu")
    if weak["dev"] != dev_ids:
        raise ValueError("Role labels and dev protocol mismatch")
    cache = PublicBatch(a.run)
    report = {}
    for arm in ("angle_only", "weak_intermediate", "weak_role"):
        state = torch.load(a.output/arm/"best.pt", weights_only=True,
                           map_location="cpu")
        model = RolePolygonBearing().cuda().eval().requires_grad_(False)
        model.load_state_dict(state["model"], strict=True)

        @torch.no_grad()
        def raw(batch, _, indices):
            return model(**batch, use_relations=True)["probabilities"]

        @torch.no_grad()
        def changed(batch, _, indices):
            alternative = dict(batch)
            phrases = batch["relation_steps"].clone()
            source = batch["relation_steps"].roll(1, dims=0)
            if len(indices) > 1:
                for i, query_id in enumerate(indices):
                    for role in weak["targets"].get(query_id, []):
                        if role["intermediate"]:
                            j, t = role["anchor"], role["step"]
                            phrases[i, j, t] = source[i, j, t]
            alternative["relation_steps"] = phrases
            return model(**alternative, use_relations=True)["probabilities"]

        base_metrics, base_rows = evaluate_cpu_metrics(
            None, cache, dev_ids, dev_labels, arm, a.batch_size, predict_fn=raw)
        fake_metrics, fake_rows = evaluate_cpu_metrics(
            None, cache, dev_ids, dev_labels, arm, a.batch_size, predict_fn=changed)
        report[arm] = dict(normal=base_metrics, intermediate_relation_shuffled=fake_metrics,
                           paired=paired_queries(base_rows, fake_rows),
                           query_with_weak_intermediate=sum(
                               any(v["intermediate"] for v in weak["targets"].get(i, []))
                               for i in dev_ids))
    atomic_json(dict(scope="TRAIN dev only", groups=report,
                     shuffle_description="only intermediate-role relation embeddings, "
                                         "rolled within each fixed evaluation batch",
                     nonlinguistic_counterfactual=True,
                     not_independent_confirmation=True,
                     no_Val_Test_RGB_navigation=True),
                a.output/"ROLE_SCOPE_RELATION_COUNTERFACTUAL_DEV.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
