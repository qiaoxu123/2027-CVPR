# Inverse Polygon Direction Belief (synthetic TRAIN-only pilot)

## Research question

Can a small text+Polygon network learn that **changing only a direction
phrase** changes the 28x28 location Belief to the appropriate opposite
side of the *same* public anchor?

This is NOT a new CityNav navigation dataset, real instruction paraphrase,
new human target, or valid full-loop navigation SR result. It is a controlled
MAP-frame synthetic geometric-direction supervision experiment.

## Inputs / truth / geometric assumptions

Source: prepared static-only CityNav TRAIN JSONL and the CityRefer TRAIN
source \`objects.json\`. Each usable query supplies a **source** target object
ID (used **offline to generate labels**) and a **public** named anchor ID.
Both must have valid, unambiguously separated source Polygon outlines.

Make one complete 4-example family:
- **Forward positive:** "Find the building map-right of One Stop."
  Input: One Stop Polygon; label: original target's Polygon.
- **Inverse positive:** "Find the building map-left of the building."
  Input: original target Polygon (synthetically promoted to the anchor);
  label: former public anchor Polygon.
- **Forward negative:** same target/anchor but replace \`map-right\` by
  \`map-left\`; **DO NOT invent new target coordinates**.
- **Inverse negative:** same swapped target/anchor but wrong direction.

The two negatives have the original target Polygon as a **negative region**
under the modified text, not a new ground-truth goal. This tests a necessary
directional property, not the ability to generate new CityNav tasks.

Crucially: The original TRAIN target contour may be used in the *inverse*
example as a newly **designated synthetic public anchor**, because its
source is allowed for offline pretraining. The original and inverse tasks
have different observable inputs and must NOT be compared as regular
benchmark episodes. A target polygon is never present in the SAME task's
forward input, and target labels are never passed to the neural forward.

Use only unequivocal MAP-relative signs:
\`map-right,map-left,map-up,map-down\`.
This DOES NOT assert true geographic east/west/north/south or camera-relative
front/back. Axis calibration is a separate prerequisite for natural-language
cardinal grounding. Diagonal, overlapping and near-boundary ambiguous pairs
are rejected rather than assigned arbitrary labels.

The source task's original description is not rewritten or mislabeled as
a human annotation. It supplies only the identity of the source target
and a valid named public Polygon. The synthetic language is template-based
and may not transfer to complex indirect CityNav instructions.

## Code

- \`graphnav/inverse_polygon_direction_data.py\`: data extraction, strict
  whole-outline direction decisions, 4-example family, TRAIN-only bounds
  and map-isolated dev split. Writes provenance, counts, hash digests.
- \`scripts/build_inverse_polygon_directions.py\`: dataset CLI.
- \`graphnav/inverse_polygon_direction_model.py\`: frozen text backbone
  conditioning, full Polygon outline encoder, candidate-to-Polygon geometry,
  small trainable 28x28 directional Belief head.
- \`scripts/train_inverse_polygon_directions.py\`: true-label soft heatmap
  CE, wrong-relation contrast and bounded divergence regularizer.
- \`tests/test_inverse_polygon_direction_belief.py\`: sample invariants,
  no positive negative label fabrication, map splits, gradients.

Model uses existing frozen \`bert-base-uncased\` for **four relation phrases**
and a trainable discrete relation adapter. This is a diagnostic vocabulary
probe; it does not yet parse free-form, long CityNav instructions.

## Generate dataset (local CityNav paths)

Use an existing \`--static-only\` TRAIN JSONL produced by
\`scripts/prepare_citynav.py\`, with map names and original source
\`episode_id=\` \`<map>:<source_object_id>:<ann_id>:0\`. Do not use
Val/Test or trajectory-sampled episodes.

\`\`\`bash
python -m scripts.build_inverse_polygon_directions \
  --train-static /path/to/graphnav_train_static.jsonl \
  --objects /path/to/cityrefer/objects.json \
  --output runs/inverse_polygon_direction_s42/data
\`\`\`

Dataset builder will fail if no trusted polygon pairs or no nonoverlapping
map-disjoint dev partition are available. Read
\`runs/inverse_polygon_direction_s42/data/inventory.json\` before training.
The full 21,878 TRAIN count is the query source size, NOT the number of
eligible directed polygon pairs. Check \`accepted_pairs\` and rejection
reasons. No claim is made that car instances all have correct Polygon labels.

## Train and evaluate

\`\`\`bash
python -m pytest -q tests/test_inverse_polygon_direction_belief.py

python -m scripts.train_inverse_polygon_directions \
  --dataset runs/inverse_polygon_direction_s42/data \
  --output runs/inverse_polygon_direction_s42/model \
  --device cuda --seed 42 --epochs 5
\`\`\`

Outputs:
- data/fit.jsonl, dev.jsonl, inventory.json
- model/last.pt, manifest.json, history.json
- model/dev_metrics.json, dev_cases.jsonl
- model/figures/pair_*.png: 2x2 matched Belief, correct forward,
  correct inverse, forward incorrect relation, inverse incorrect relation.

\`--resume\` resumes the same checkpoint, split, frozen BERT,
hyperparameters. Do not use val-seen for checkpoint selection.

## Required acceptance checks

1. **Pair integrity:** opposite relation sign on true inverse task;
   the original target never appears as a forward model input; all
   negative statements preserve the same target label, only truth flips.
2. **Direction responsiveness:** With the SAME anchor Polygon, change
   only the direction relation. Compute target Belief centroid for both
   versions; \`changed_forward_toward_new_direction_m\` and
   \`changed_inverse_toward_new_direction_m\` must be POSITIVE.
   Report success fractions and displacement magnitudes over ALL dev pairs.
   The word must actually change the model output; a static geometry
   prior alone is insufficient.
3. **Held-out map:** report synthetic positive proxy_NE_m and the
   pairing/word response metric. Compare with a simple analytical
   polygon-direction prior and a model with direction embeddings shuffled
   at inference, so an easy template task cannot masquerade as an
   improvement in real instruction understanding.
4. **Main benchmark:** DO NOT use this synthetic positive proxy_NE_m
   as CityNav navigation error, Top-1 SR or closed-loop SR. To test whether
   the module improves original instructions, implement a separately
   calibrated, GT-free natural-language axis adapter and rerun TRAIN-Dev
   original-query paired Geometry baseline before any Val-Seen closed loop.

Go/no-go:
- no held-out map counterfactual movement => do not integrate;
- obvious synthetic movement but no improvement on original-query
  R@20m and no safe cardinal frame => keep as geometry pretraining only.
