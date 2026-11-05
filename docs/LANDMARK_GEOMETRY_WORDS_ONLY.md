# CityNav minimal: public landmark name + adjacent geometric words

This is deliberately NOT the full Geometry+Relation+RGB system. No target
type, color, appearance, intermediate object, relation graph, full-sentence
BERT, original pretrained Geometry features, RGB, teacher trajectories or
GT-selected landmark IDs reach model forward.

| Full original description (used only by lexical extractor) | Landmark | Geometry words |
|---|---|---|
| Blue car behind One Stop | One Stop | behind |
| Yellow building north of One Stop | One Stop | north of |
| White house next to Blackwell Street | Blackwell Street | next to |
| Car in parking lot behind One Stop | One Stop | behind |
| Car near unnamed parking lot | none | none |

Important: "parking lot behind One Stop" is a lexical relation to the
landmark, NOT evidence the *target car* is directly behind it. Ambiguous
reference frames behind/left/right are not hardcoded as world coordinates.

## Code

- graphnav/landmark_geometry_words.py: use public name spans + adjacent
  exact spatial phrase only. Keep all source IDs sharing the same visible
  name; reject overflow and invalid Polygon geometry explicitly.
- graphnav/landmark_geometry_only_model.py: newly trained 28x28 Goal Belief
  based on frozen BERT landmark-name embedding, frozen BERT geometric-phrase
  embedding, and 9d candidate-to-PUBLIC-Polygon geometric features.
  TRAIN FROM SCRATCH; no pretrained Geometry hidden state as shortcut.
- scripts/train_landmark_geometry_words.py: all 21,878 Revised TRAIN tasks
  accounted (Cambridge fit; Birmingham TRAIN-Dev); seed42, 3 epochs.
  All tasks including unknown names stay in denominator. Empty input
  predicts a uniform Belief.
- tests/test_landmark_geometry_words_only.py: extraction, no-leakage, gradients.
- Output: manifest.json, extraction_audit.jsonl, last.pt,
  history.json, report.json.

Unlike the preselected TRAIN landmark list (originally produced through
target-specific processed annotations), model inputs come solely from
explicit name matching against the FULL public CityRefer objects.json.
Original text is used only for lexical extraction; target GT used only
AFTER forward to calculate training/evaluation loss.

## Commands: run on the CityNav server

```bash
cd /home/20261024_1/Workspace/2027-CVPR/Experiments/GeomBeliefLandmarkChain
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python
BASE=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_full_relation_s42_20261105
OBJECTS=/home/20261024_1/Workspace/DATA/sbfnav-data/cityrefer_original/objects.json

"$PY" -m pytest -q tests/test_landmark_geometry_words_only.py

# First smoke only (not full TRAIN coverage).
"$PY" -u -m scripts.train_landmark_geometry_words \
  --train-jsonl "$BASE/train_revised.jsonl" \
  --public-objects "$OBJECTS" \
  --output "$BASE/landmark_words_smoke_s42" \
  --epochs 1 --batch-size 16 --seed 42 \
  --max-train-batches 12 --max-dev-batches 12

# Full, separate output: every 21,878 task included in source inventory.
"$PY" -u -m scripts.train_landmark_geometry_words \
  --train-jsonl "$BASE/train_revised.jsonl" \
  --public-objects "$OBJECTS" \
  --output "$BASE/landmark_words_full_s42" \
  --epochs 3 --batch-size 16 --seed 42
```

Resume only from completed epoch using --resume and exactly matching args.
Do not use Val-Seen/Test or closed-loop during this experiment.

## Interpret results

Report all-task and pair-present subsets separately: R@20m, NE,
corrections/damages versus name-only, comparison with scrambled spatial
phrases, matched names/phrases/no-public-name coverage, and failure cases.
Do not claim semantic target-to-anchor directness from text adjacency alone.
High R20 is not expected for instructions where many target objects share
the exact same public landmark and geometric phrase; this input deliberately
discards the target identity. Historical Geometry+Relation results are
not apples-to-apples (they used full instruction and pretrained models).


## Executed experiment (2026-11-05)

See `reports/landmark_geometry_words_only/REPORT_zh.md` and `index.html`.
All 21,878 rows accounted: 14,259 Cambridge fitting, 7,619 Birmingham held out.
Seed42, three complete epochs, newly initialized 177,601-parameter model;
no original Geometry/Relation/RGB checkpoint loaded. All comparisons are
same-final-model input ablations, not separately trained best models.

- Geometry phrases are globally permuted within held-out data at query/name
  level, preserving same-name Polygon-part agreement and the original word
  mask. Results do not depend on evaluation batch size.
- Raw invalid/zero-area polygons are rejected; compound cardinal phrases and
  directly attached `not` retain their exact relation text.
- Epoch checkpoint saves optimizer/model/RNG; restore was tested without
  changing any original experiment output.
- Correct words R20 9.765%, name-only 9.148%, shuffled words 9.896%: no
  demonstrated semantic advantage over shuffled phrases; no navigation claim.
