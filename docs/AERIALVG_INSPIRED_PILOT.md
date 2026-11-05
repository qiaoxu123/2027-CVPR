# AerialVG-inspired CityNav relation experiment (independent implementation)

> **Superseded for full experiments:** the 573-query standalone pilot below
> remains a diagnostic. The integrated all-instruction UnifiedPGVB pipeline,
> full TRAIN and complete Val-Seen commands are in
> [UNIFIED_FULL_RELATION_TRAIN.md](UNIFIED_FULL_RELATION_TRAIN.md).

**Status:** experimental, TRAIN-only. Not an implementation or port of the
AerialVG authors' pretrained model, and not yet integrated into normal CityNav
inference/Stop. Preserves the parent PR #16 behavior.

## Motivation

AerialVG (ICCV 2025) uses target/reference candidate pair representations
and alternating text cross-attention and relation self-attention.
Reference: https://github.com/Ideal-ljl/AerialVG/blob/main/model/AerialVG/relation.py

Our independent CityNav-native prototype adapts the *principle*:
- Target: every one of the 28x28 Geometry Belief grid cells.
- Reference: each public named anchor Polygon (no GT binding).
- Pair features: existing 128-d Geometry dense context for target/reference,
  10 raw public Polygon geometry features, 8 predicate IDs.
- Language: existing frozen 768-d BERT token cache.
- Relation: learned pair embedding, text cross-attention, inter-anchor
  self-attention, score aggregation, bounded dense Belief logit residual.
- Zero-initialized score head: identical Geometry baseline before training.
- No additional image model, hand-authored inside/outside masks, or semantic
  neighborhood threshold. Does **not** solve missing functional-region anchors
  without better candidate data. Relations can be wrong if binding is wrong.

## Local reproducible TRAIN experiment

Use your existing current PR #16 573-query TRAIN inventory, static/visual cache,
and trained direct-target checkpoint. This is a strict relation pilot, **not**
a full 22k-task CityNav training run:

```sh
python -m pytest -q tests/test_aerialvg_inspired_relation.py
python scripts/train_aerialvg_inspired_relation.py \
  --inventory /path/to/train-inventory \
  --cache /path/to/train-cache \
  --checkpoint /path/to/unified-direct-target/best.pt \
  --output runs/aerialvg_inspired_train_s42 \
  --epochs 4 --batch-size 8 --seed 42
```

Outputs: `last.pt`, `summary.json`, `dev_cases.json`. Evaluate paired
Geometry vs learned relation R@20m/NE, fixes/breaks on **TRAIN dev only**.
The full baseline and original checkpoint remain immutable.

## Decision

Do not change the online policy or run Val-Seen until:
1. TRAIN-dev R@20m and NE improve, with fewer breaks than fixes.
2. Counterfactual relation changes lead to correct Belief movements.
3. Target/reference pair selection succeeds on independently checked examples.
4. There is no GT or future view entering the forward inputs.

Later tests should compare the original pretrained AerialVG independent detector
with our original relation head. The optional `aerialvg_pretrained_bridge.py`
only maps predicted boxes to CityNav world XY and is not the main experiment.
