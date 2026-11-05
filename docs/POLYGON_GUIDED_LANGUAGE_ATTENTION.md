# Polygon-Guided Language Attention (PGLA): lightweight CityNav TRAIN experiment

**Status:** new standalone module, TRAIN-only training/evaluation. NOT a
deployed navigation policy; no new Val-Seen/Val-Unseen/Test or SR claim.
Original trained Geometry+Relation, RGB, Stop and controller stay unchanged.

## Motivation

Do not hard-code a larger weight for "north", "behind" or "right": the
**relation phrase + referenced landmark/region** determines its usefulness.
For each candidate cell on the 28x28 map, compute:

1. public candidate-to-landmark Polygon geometry (existing [B,Q,N,10]);
2. candidate-weighted public entity/name geometry context (existing BERT names);
3. **candidate-specific attention** over full BERT instruction wordpieces,
   excluding [CLS], [SEP] and padding;
4. bounded logit residual on top of the **frozen completed
   Geometry+Relation checkpoint**. No RGB, graph parser, new labels or GT
   inputs during forward.

`delta = .5 * tanh(MLP(geometry_context, BERT_word_context, base_dense))`.
Final layer is zero-initialized so initial prediction equals the baseline.
No public anchor => exact zero delta. This is a lightweight candidate
reranker, not a learned world direction system.

**Attention score != causal word importance.** Top-weighted tokens are only
diagnostic. To prove the model uses spatial terms, test realistic
relation-phrase substitutions in the same public anchor scope and verify
changes in Belief, then check the valid geometric direction when the
reference frame is independently grounded. "left/right/behind" are NOT
automatically world east/west/north/south.

## Reuse

- `graphnav.polygon_language_attention.PolygonGuidedLanguageAttention`
  standalone trainable residual, no changes to existing model weights.
- `graphnav.candidate_geometry.FullCandidateDataset` 28x28 world grid,
  public polygons and per-candidate 10d metric geometry.
- `graphnav.full_relation_training.load_full_relation_model`, completed
  full-relation checkpoint; freeze Geometry and Relation.
- `graphnav.token_language.FrozenBERTTokens` full wordpieces + public names.
- `scripts.train_polygon_language_attention` deterministic seed=42,
  3 epochs by default, Cambridge TRAIN-Fit, Birmingham TRAIN-Dev. Frozen
  base itself previously trained on full TRAIN, so this is a probe of the
  NEW head's fit/dev transfer, NOT an independent unseen-scene proof.

## Commands (run on local CityNav GPU server)

Use the existing full-relation source and checkpoint from
`reports/unified_pgvb/full_relation_s42_20261105/final_evaluation.sh`.

```bash
cd /home/20261024_1/Workspace/2027-CVPR/Experiments/GeomBeliefLandmarkChain
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python
RUN=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_full_relation_s42_20261105

"$PY" -m pytest -q tests/test_polygon_language_attention.py

# Fast training/Dev SMOKE. Do NOT call these partial results full TRAIN-Dev.
"$PY" -u -m scripts.train_polygon_language_attention \
  --train-jsonl "$RUN/train_revised.jsonl" \
  --relation-checkpoint "$RUN/core/last.pt" \
  --output "$RUN/polygon_language_smoke_s42" \
  --epochs 1 --batch-size 8 --seed 42 \
  --max-train-batches 12 --max-dev-batches 12

# Full, separate output. No --max-... flags.
"$PY" -u -m scripts.train_polygon_language_attention \
  --train-jsonl "$RUN/train_revised.jsonl" \
  --relation-checkpoint "$RUN/core/last.pt" \
  --output "$RUN/polygon_language_full_s42" \
  --epochs 3 --batch-size 8 --seed 42

# Direction phrase swap: fixed Polygon and base, only BERT tokens change.
"$PY" -u -m scripts.probe_polygon_language_words \
  --train-jsonl "$RUN/train_revised.jsonl" \
  --relation-checkpoint "$RUN/core/last.pt" \
  --adapter-checkpoint "$RUN/polygon_language_full_s42/last.pt" \
  --output "$RUN/polygon_language_full_s42/word_swap_probe" \
  --max-pairs 100
```

`--resume` restarts only at a completed-epoch boundary; does NOT restart
a partially completed epoch. Any changed source data or model code makes
resume refuse the existing checkpoint manifest.

## Required checks

Read `report.json` with:
- Frozen `static` Geometry+Relation baseline;
- `real` actual instruction tokens;
- `zero_text` same geometry/Polygon/name features but instruction tokens 0;
- `swapped_text` identical geometry, token features from a *different*
  instruction in batch. Do NOT shuffle token order within the SAME sentence:
  standard attention is permutation invariant over keys/values, so that
  would NOT test text relevance.
- TRAIN-Dev R20, NE, corrected/broken and Top1 changes for all tasks;
  `example_token_attention` only qualitative, not an explanation proof.
- Never use target GT as a mask, pseudo key-word label, candidate crop or
  selector during forward.
- Keep zero/invalid anchors in original denominator.
- No checkpoint selection by Dev performance; inspect final epoch only.
- Stop before any Val-Seen closed loop unless the true-language head
  demonstrably beats the static baseline AND zero/swapped text controls
  in held-out TRAIN-Dev, without an excessive damage count.

## Next check before navigation

On held-out TRAIN examples, perform **scope-checked** spatial-phrase swaps
(e.g. north/south of the SAME public landmark) with the other words held
constant. For direction metric use only independently verified north/world
frame. For left/right/behind, first identify whose view defines the direction.
Re-encode the modified instruction and recompute the language residual
with identical frozen base/Polygon to isolate the new head. A high attention
weight without matching counterfactual Belief response is not evidence
that the word was meaningfully learned.

The new head is NOT loaded by the old closed-loop checkpoint loader.
Keep this separate experiment branch until trained and validated.

The word-swap probe outputs `summary.json` and `pairs.jsonl`. These
measure language-head sensitivity, not automatically correct direction.
Do NOT set `--verify-world-north` without independent CityNav map CRS
orientation verification. Ambiguous behind/left/right are skipped.
