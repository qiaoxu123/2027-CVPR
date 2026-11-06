# HRGB: Explicit Hierarchical Relation Graph (TRAIN-only pilot)

**Status: source committed, NOT executed, trained, benchmarked or integrated into
the online navigator.** This experiment is deliberately isolated from the
verified Geometry/PGVB/Trajectory/Stop runtime.

## What is actually implemented

- Every one of the current **28 x 28 = 784 dense goal cells** is a fine target
  candidate. These are not vehicle-instance proposals or target-GT crops.
- Public **named reference Polygon nodes** form the landmark level.
- When the English instruction unambiguously describes a vehicle *in a parking
  lot*, public Parking polygons from the official CityRefer objects map form
  the intermediate region level. Region selection uses the reference
  polygons' bounding boxes and public object identity, never target coordinates.
- Geometry is **metric**, in the same map world coordinates. Existing
  candidate-to-anchor [Q,A,10] geometry is reused; candidate-to-Parking [Q,R,10]
  and Parking-to-anchor [R,A,10] use complete published contour geometry through
  relative_entity_features.
- Language-conditioned attention runs landmark->region->fine candidates;
  a **separately inspectable** explicit distance/containment likelihood is
  added to the learned residual.
- A spatial softmax gives a goal-cell distribution. Non-grounded instructions,
  missing references, or missing required parking regions *abstain*, preserving
  the exact frozen Geometry field. No external vision model or extra RGB is used.

This is a deliberately **narrow, falsifiable two-hop proof-of-concept**:
it does **not** yet handle parking-row instances, unknown parking boundaries,
free-form indirect relations, 3D view synthesis or target-object proposals.
The parser leaves uncalibrated left/right/behind without a geometric direction.
As in earlier experiments, public CityRefer polygons are weak geometry, not
independent instance detection ground truth.

## Frozen source and reproducibility

- Base source commit: 4d195f841ba9c3a561555ca75bcf44bfea79b6cc.
- Original unchanged checkpoint: verified UnifiedPGVB-v2
  (bundle format unified-pgvb-v2).
- All data rows are **Revised TRAIN static-only**, with 0 future RGB, GT crops,
  robot poses or teacher futures passed to the forward path.
- TRAIN-only fit/dev partition is hashed by physical target ID so the same
  physical target cannot fall on both sides. This is not independent map-level
  generalization, because both groups still share city scenes.
- The first 2000 TRAIN rows are a fixed convenience pilot, not balanced direct
  and indirect sampling. Report exact numbers of active/unresolved examples.

## Run on the local GPU machine

Install the existing package with its foundation and dev requirements:

    pip install -e ".[dev,foundation,data]"

Run small CPU regression tests first:

    python -m pytest -q tests/test_hierarchical_relation_graph.py

Execute a first smoke using TRAIN data only (it still trains, not a dry-run):

    python scripts/train_hierarchical_relation_graph.py \
      --train-jsonl data/unified_full_train_revised.jsonl \
      --objects /path/to/CityNav/cityrefer/objects.json \
      --original-v2 GraphNav/runs/unified_pgvb_v2_pilot_s42/full_training/last.pt \
      --output runs/hrgb_smoke_s42 \
      --max-queries 128 --epochs 1 --batch-size 4 --seed 42

Then run the main TRAIN-only paired evaluation with independently
specified output directory:

    python scripts/train_hierarchical_relation_graph.py \
      --train-jsonl data/unified_full_train_revised.jsonl \
      --objects /path/to/CityNav/cityrefer/objects.json \
      --original-v2 GraphNav/runs/unified_pgvb_v2_pilot_s42/full_training/last.pt \
      --output runs/hrgb_full_s42 \
      --max-queries 2000 --epochs 4 --batch-size 4 --seed 42

Paired conditions on the **same frozen Geometry, TRAIN rows and training budget**
(the controls require separate fits and output directories):

- --ablation full : direct + explicit Parking two-hop graph.
- --ablation flat : direct anchor-to-cell graph; ignores intermediate Parking.
- --ablation no-geometry : learned graph without numeric edge geometry.
- --ablation no-language : removes BERT embedding but intentionally retains
  the conservative explicit relation route/type. It is a no-BERT control, **not**
  a genuinely language-free baseline.

Results are written to last.pt, summary.json and dev_cases.json. They are
**offline** Geometry vs Hierarchy Top-1 R@20m, NE, corrections/breakages,
GT-neighborhood belief mass and mean logit change, with active direct, active
indirect and unresolved counts. Never describe these as closed-loop SR/OSR.

## Acceptance and limitations

1. Require all tests to pass, including permutation invariance of reference
   nodes and intermediate regions; absent intermediates must preserve Geometry.
2. Confirm the exact checkpoint and input SHA, that no Val/Test rows were read,
   and that no GT, target ID or future observation was used to select a region.
3. Report per-layer support. If there are too few indirect examples, do not
   claim a general result for indirect relations. Repeat with a **fixed,
   predeclared** TRAIN-only indirect subset; no GT-guided sample selection.
4. Compare paired R@20m, NE, fix/break count, belief motion and confidence
   under the same sample denominators. A larger logit change does not itself
   demonstrate better relational reasoning.
5. If no clear TRAIN-dev benefit, STOP: do not wire the module into the online
   policy, add layers, tune Val-Seen or change Stop thresholds.
6. If there is a reproducible TRAIN-dev improvement, the next stage is a
   **separately reviewed** candidate-instance graph and controlled online
   integration. This branch makes no claim about improved navigation SR.

The original Geometry, Relation, PGVB, Stop and online evaluator are unmodified.
