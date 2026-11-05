# UnifiedPGVB full CityNav + AerialVG-inspired relation: execution

**Status:** an experimental full-instruction integration on PR #17.
**Not** official AerialVG model weights. **Not yet GPU-validated or an SR claim.**

## Full-sample interface

- Full original instruction BERT tokens are passed directly to Geometry and
  candidate-to-public-anchor relation attention. No
  `select_direct_relations` eligibility filter.
- For each target grid cell, public candidate entities are scored as potential
  references using their relative 10-D contour geometry. A learned null
  fallback retains Geometry for queries with no matched reference.
- Relationship words are understood through full BERT cross-attention rather
  than a strict eight-class predicate parser or hard inside/near mask.
- Relation scores are added to the same 28x28 Belief *before* Top-K and
  passed to the unchanged trajectory/Stop controller.
- The existing PGVB branch operates only on CURRENT observed RGB, and only
  when there is a uniquely identified public visual anchor with the original
  consecutive-frame discriminative gate. Ambiguous/no-anchor instructions
  still run Geometry+Relation. RGB does not independently verify target ID.
- Geographic N/S/E/W is not labeled by an unverified World XY orientation.

## Preflight: source completeness

From repo root, using the same Revised CityNav source and original model
paths as PR #16:

```sh
python -m pytest -q tests/test_aerialvg_inspired_relation.py tests/test_unified_full_relation.py
python scripts/prepare_citynav.py \
  --objects "$DATA_ROOT/cityrefer/objects.json" \
  --processed "$DATA_ROOT/cityrefer/processed_descriptions.json" \
  --trajectories "$DATA_ROOT/processed_citynav/citynav_train_seenall.json" \
  --bounds "$DATA_ROOT/citynav_bounds.json" \
  --output "data/unified_full_train_revised.jsonl" \
  --dataset-version revised --match-mode fuzzy --static-only
```

Review the `episodes_skipped` count and `records_generated` in its
`.provenance.json`. The converter may skip malformed original records; the
full-train claim must state original total, converted total and rejected total.
Unrecognized names, indirect relationships, missing polygons and empty
references are **not** parser-based exclusion reasons.

## Phase 1: full core training on ALL converted TRAIN instructions

```sh
python scripts/train_unified_full_relation.py \
  --train-jsonl data/unified_full_train_revised.jsonl \
  --stage1 "$STAGE1" --original-v2 "$UNIFIED_V2" \
  --output runs/unified_full_relation_s42 --epochs 5 \
  --batch-size 8 --seed 42
```

To resume after an interrupted epoch, repeat with `--resume`. Check
`manifest.json`, per-epoch expected/observed counts, `progress.json` and
`last.pt`. For a new GPU smoke use a **different** output directory and
`--max-batches 2`; smoke checkpoints are **not** full trained models.

Phase 1 jointly trains the existing Geometry model and NEW relation head on
complete TRAIN static queries. Frozen original BERT is used as the language
feature extractor. Pretrained PGVB weights are kept unchanged in this phase
because static-only records have no current-frame visual tensor.

## Optional phase 2: train existing PGVB with real observed TRAIN RGB

This stage uses the previously prepared actual-RGB direct-target TRAIN cache,
not invented global orthomosaic target crops:

```sh
python scripts/train_unified_full_visual.py \
  --base-checkpoint runs/unified_full_relation_s42/last.pt \
  --inventory "$DIRECT_TRAIN_INVENTORY" \
  --cache "$DIRECT_TRAIN_CACHE" \
  --output runs/unified_full_relation_rgb_s42 \
  --epochs 3 --batch-size 4 --seed 42
```

Geometry and learned relation are frozen during phase 2 to preserve full
TRAIN coverage learning; PGVB is trained on the available causal RGB subset.
This visual subset is smaller and biased toward direct relations. Report
`visual_rows` and `ineligible_visual_rows` and compare results with phase
1. Do not claim visually full-sample training or full end-to-end SigLIP2
fine-tuning.

## Phase 3: ALL val-seen closed-loop, unchanged controller

The primary evaluator now accepts `--model unified-relation`. No
`--max-episodes` is permitted. Two examples for matched ablations:

```sh
python -m scripts.evaluate_closedloop \
  --model unified-relation --split val_seen \
  --sbf-source "$SBF_SOURCE" --config "$SBF_CONFIG" \
  --data-root "$DATA_ROOT" --checkpoint "$STAGE1" \
  --unified-v2 "$UNIFIED_V2" \
  --relation-checkpoint runs/unified_full_relation_s42/last.pt \
  --relation-control off --relation-visual-control off \
  --output runs/unified_full_relation_s42/eval_geometry_full

python -m scripts.evaluate_closedloop \
  --model unified-relation --split val_seen \
  --sbf-source "$SBF_SOURCE" --config "$SBF_CONFIG" \
  --data-root "$DATA_ROOT" --checkpoint "$STAGE1" \
  --unified-v2 "$UNIFIED_V2" \
  --relation-checkpoint runs/unified_full_relation_s42/last.pt \
  --relation-control full --relation-visual-control tracked \
  --output runs/unified_full_relation_s42/eval_combined_full
```

For phase-2 visual weights replace `--relation-checkpoint` with
`runs/unified_full_relation_rgb_s42/last.pt`. To resume a partial evaluation,
add `--resume` and use the **same** output directory and immutable inputs.

Run at least a third paired condition `--relation-control full
--relation-visual-control off` to separate RAGM from observed PGVB.
Compare SR/OSR/SPL/NE/nDTW, no-reference tasks, inference errors (always
counted as failures), candidate switches and observed RGB activation.
The full original benchmark denominator includes **every CityNavDataset
Val-Seen episode**; neither empty candidate nor parsing uncertainty is a
reason to drop an episode.

## Promotion checklist

1. CPU CI + GPU smoke + exact protocol/weight SHA checks pass.
2. The per-epoch TRAIN count matches converted source; provenance documents
   malformed upstream skips instead of silently claiming 100% raw coverage.
3. Three full Val-Seen ablations are completed on the **same episodes**.
4. Report both gains and damage, with seed 42 and no Val/Test tuning.
5. Only then consider replacing the official main experiment.

## Server execution notes (2026-11-05)

- Available Revised source is `processed_citynav/citynav_train_seen.json`:
  **21,878 records converted, zero skipped**. Do not substitute a missing
  `citynav_train_seenall.json` path or a 573-query selection.
- Public bounds were read from the corresponding TRAIN raster transforms;
  no private destination was used to derive map bounds.
- `--resume` now restores deterministic within-epoch order, optimizer and RNG
  from periodic 200-update checkpoints. BERT feature cache is reused.
- Phase-2 entries require the same supported local Polygon ROI as runtime.
  Rejection statistics explicitly distinguish query exclusions from frame
  exclusions; they are not all independently sampled images.
- All-episode evaluator stores step Beliefs/RGB SHA and retains torn final
  JSONL bytes for recovery. Stop and Goal constants are unchanged.
- Optional `--relation-geometry-source original` is legal only with relation
  and visual controls both off. This adds a full frozen baseline; the three
  main ablations continue to share the trained Geometry weights.
