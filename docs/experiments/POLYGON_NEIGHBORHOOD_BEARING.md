# CityNav: Neighborhood Polygon Bearing Predictor

## Scope

This experiment compares two **matched** 8-bin bearing heads trained from CityNav TRAIN target-angle labels:

- `text_name_control`: complete BERT instruction tokens and named landmark text.
- `public_polygon_neighborhood`: same text, plus each named anchor's shape and up to 24 nearest public Polygon shapes/offsets within 150 m.

The arms use the same head architecture, seed 42, map-disjoint split, optimizer, and supervision. The only ablation is whether the public geometry is visible.

**GT is never forwarded into the model.** It is used only to construct 8-bin target-angle labels and held-out error metrics. References and neighborhoods are selected using original language and the public Polygon catalog. This does not recover annotator yaw or prove that arbitrary anchor->GT angles are expressed by the sentence.

Unlike previous PR25 experiments, this adds **all publicly mapped nearby polygons** (not merely named anchors and OSM roads). No RGB, local frame rules, Controller changes, navigation rollouts or SR claims.

## Files

- `graphnav/polygon_neighborhood_bearing.py`: GT-free public neighborhood and relation-aware direction head.
- `scripts/train_polygon_neighborhood_bearing.py`: full TRAIN matched-pair fit and held-out-map evaluation.
- `tests/test_polygon_neighborhood_bearing.py`: CPU regression tests for geometry sensitivity, no GT feature access, masking, and map isolation.

## Run on the existing CityNav GPU machine

```bash
python -m pytest -q tests/test_polygon_neighborhood_bearing.py
python -m scripts.train_polygon_neighborhood_bearing \
  --train-jsonl /path/to/train_revised.jsonl \
  --existing-full-checkpoint /path/to/unified_full_relation/last.pt \
  --public-objects /path/to/cityrefer_original/objects.json \
  --output /path/to/new_run_dir \
  --epochs 3 --batch-size 8 --seed 42 --device cuda
```

Default data inventory is 21,878 Revised/fuzzy TRAIN tasks. About 20% of the maps are held out deterministically (seed 42); override with `--heldout-maps mapA mapB` only before evaluating. No Val/Test data is used. Smoke-only `--max-train-batches 3 --max-eval-batches 2` must not be presented as full experimental results.

Outputs: `REPORT.json`, each arm's heldout rows JSONL, training histories/checkpoints, and `protocol.json` with SHA-256 input provenance.

Metrics: Top-1/Top-2 of 8 bins, circular angular MAE and median, NLL, all/single/multi-anchor breakdown and per-map results. Each TRAIN query contributes equally to loss even with multiple referenced polygons.

## Important caveats

1. Historical 41.13% was a TRAIN statistic, not map-heldout accuracy. Compare the two arms **within this matched study**.
2. Full catalog may naturally contain the target object as an unlabeled public polygon, but neighbor selection never uses its identity or GT position. The study remains a statistical target-bearing probe, not a clean intrinsic-frame ground truth dataset.
3. If one anchor's direction is not semantically identifiable, the correct geometric GT label need not be linguistically predictable. A low score may reflect underdetermination rather than optimizer failure.
4. The old head, runtime/controller and main GeomBelief files remain unchanged. This code does **not** yet produce measured accuracy until the local dataset and GPU run complete.
