# Full-Token Graph Geometry + Observed RGB: CityNav A0–A5

This branch is based on \`2027-CVPR/full-landmark-geometry-grounding\`. **It does not reuse HETT's network**. It directly addresses differences observed against the locally reproduced SBF selector. No SR/SPL results are claimed until the CityNav simulator is attached.

## Why change S2?

Previous S2 only consumed BERT CLS and semantic relation rules for candidate scoring. It did not receive per-candidate/per-landmark boundary distance, signed east/north displacement or whole BERT token states. Further, comparing *Original CityNav strict names* to *Revised CityNav fuzzy names* confounds architecture with dataset and binding policy.

## Network (all 784 cells scored before Top-K)

\`\`\`text
               Whole BERT instruction token sequence [L,768]
                                    |
                           Cross-Attention Q=784
                                    |
Candidate world XY [784,2] ---> candidate token states [784,H]
                                    |
Independent source ID + Named Polygon geometry + object_type
             |                      |
             +--- typed Edge-GATv2 --+   (A4 only)
                       |
    each candidate -> EACH named polygon (NOT anonymous centroid)
    Δeast/Δnorth meters, d_center, unit bearing, signed d_boundary,
    inside polygon, shape bbox, true polyline bit [Q,N,10]
                       |
              Geometry-Aware Attention (null node safe)
                       |
              28×28 Global Goal Logits
                       |
              Top-K=20 predicted candidates
                       |
     Only previously UAV-observed RGB, footprint + actual valid-pixel mask
                       |
         Frozen SigLIP2 image feature (no full-map orthomosaic)
                       |
              Observed-only gated RGB residual
                       |
      Updated *global* 28×28 logits, R@1/R@4/R@8/20m, NE
\`\`\`

Language attention is **token-level** (not CLS-only). Geometry is **candidate-specific and landmark-specific**, including real polygon boundary distance, compass and containment. Typed-GAT adds entity-to-entity information on top of the same candidate geometry. Building, TrafficRoad, Footpath, Bridge, Parking and Ground remain whole **closed polygon** nodes; real connected OSM roads are separate open polyline nodes. GAT still does not invent topology for area road surfaces.

## Fair ablations

| Run | Inputs / contribution | Training |
|---|---|---|
| A0 \`building_prior\` | Only building landmark spatial prior | fixed |
| A1 \`prior\` / \`rules\` | All source-type land polygons / plus explicit geometry rules | fixed |
| A2 \`tokens\` | Full BERT token cross-attention with 784 spatial cells | independent weights |
| A3 \`geometry\` | A2 + individual candidate-to-contour geometry | independent weights |
| A4 \`gat\` | A3 + typed Edge-GATv2 message passing | independent weights |
| A5 \`rgb\` | A4 checkpoint + actually observed SigLIP2 candidate evidence | only visual residual trainable |

Training \`tokens\`, \`geometry\`, and \`gat\` from **separate random initializations** isolates their network mechanism better than evaluating untrained heads. All are trained using the same target distribution and split. A5 starts from Val Seen-selected A4, freezes the A4 backbone and trains **only the visual residual**; this is a causal controlled comparison. Goal GT and teacher trajectories NEVER enter forward(); RGB candidates are selected from model predictions (not oracle GT candidate regions). Baseline logits are untouched when no candidate is observed.

This is an **offline localization** protocol, not a closed-loop navigation result. Candidate RGB evidence is observational; without reliable instance matching it should NOT be called an exact named-landmark visual match. Named landmark verification is a separate optional module (\`visibility_graph.py\`) requiring independently grounded matched IDs.

## Revised CityNav and matching

**YOU MUST validate locally** that the four JSON source assets match the exact Revised CityNav revision used in the local SBF reproduction. This repository CANNOT auto-detect a revision simply from its folder name. The conversion writes a \`*.provenance.json\` with SHA256 of \`objects\`, \`processed\`, \`trajectories\`, \`bounds\`. Compare those hashes with the SBF experiment's source files. If schemas differ, adapt the converter to the actual Revised fields; never silently fall back to Original.

Three name binding protocols:
- \`--match-mode exact\`: canonical exact name match only
- \`--match-mode typed\`: exact first, then type-compatible suffixes
- \`--match-mode fuzzy\`: typed fallback, then conservative normalized-string similarity with type restrictions and runner-up ambiguity margin; **no GT nearest-object selection**
Match policy and dataset version are attached to every row. The evaluator rejects train/seen/unseen files with mixed versions/policies. For equal SBF comparison, inspect the LOCAL SBF fuzzy-name matcher and align thresholds/collision handling; the provided fuzzy matcher is an intentionally conservative implementation, **not an assertion of bitwise parity with SBF**.

Use one data-version/matcher combination at a time and report its anchor/relation coverage. \`--dataset-version revised\` is an explicit declaration, **not a source schema conversion utility**.

## Data preparation

For Original or Revised (after checking the local JSON schema) replace all placeholders with verified files from the SAME revision:

\`\`\`bash
git fetch origin
git checkout 2027-CVPR/full-token-geometry-observed-rgb
pip install -c configs/static_geometry_rtx6000_constraints.txt -e '.[dev,foundation]'
pip install rasterio

# First: A0–A4 static-only. Run for train_seen, val_seen, val_unseen
python -m scripts.prepare_citynav \
  --objects "$CITYNAV_DATA/cityrefer/objects.json" \
  --processed "$CITYNAV_DATA/cityrefer/processed_descriptions.json" \
  --trajectories "$CITYNAV_DATA/processed_citynav/citynav_train_seenall.json" \
  --bounds data/citynav_bounds.json \
  --dataset-version revised --match-mode fuzzy --static-only \
  --output data/graphnav_train_seen.jsonl

# Repeat with corresponding val_seenall/val_unseenall trajectory source.
# The three files must all come from the same CityNav revision.
python -m scripts.prepare_static_splits \
  --data-dir data --refresh-raw \
  --output runs/fulltoken/query_deduplication.json

python -m graphnav.full_experiment \
  --train data/graphnav_train_seen.jsonl \
  --val-seen data/graphnav_val_seen.jsonl \
  --val-unseen data/graphnav_val_unseen.jsonl \
  --dataset-version revised --match-mode fuzzy \
  --variants tokens geometry gat \
  --device cuda --batch-size 16 --epochs 5 \
  --output runs/fulltoken_static
\`\`\`

### RGB experiment — no future observations

For A5, **reconvert all three splits** with \`--static-only\`, \`--rgbd-dir\`,
\`--observations-out\`, \`--ground-levels\` and \`--rgb-observation-step\`.
Use \`--rgb-observation-step 0\` first: only the initial UAV view from the human trajectory is exposed, not later teacher visits. Larger cutoffs are allowed for a clearly-labeled *teacher-observation-prefix diagnostic* only, and **must not** be described as policy-closed-loop RGB.

\`\`\`bash
# Example: train_seen; repeat identically for val_seen and val_unseen
python -m scripts.prepare_citynav \
  --objects "$CITYNAV_DATA/cityrefer/objects.json" \
  --processed "$CITYNAV_DATA/cityrefer/processed_descriptions.json" \
  --trajectories "$CITYNAV_DATA/processed_citynav/citynav_train_seenall.json" \
  --bounds data/citynav_bounds.json \
  --ground-levels data/ground_levels.json \
  --rgbd-dir "$CITYNAV_DATA/rgbd" \
  --observations-out data/observed_revised/train_seen \
  --rgb-observation-step 0 \
  --dataset-version revised --match-mode fuzzy --static-only \
  --output data/graphnav_train_seen_rgb.jsonl

# After converting all three RGB variants, use a separate directory and
# rename/copy them to data/graphnav_{split}.jsonl in that directory. Then
# apply prepare_static_splits --refresh-raw to the new RGB directory.

python -m graphnav.full_experiment \
  --train data/revised_rgb/graphnav_train_seen.jsonl \
  --val-seen data/revised_rgb/graphnav_val_seen.jsonl \
  --val-unseen data/revised_rgb/graphnav_val_unseen.jsonl \
  --dataset-version revised --match-mode fuzzy \
  --variants tokens geometry gat rgb \
  --device cuda --batch-size 16 --epochs 5 --rgb-epochs 3 \
  --min-visual-coverage 0.30 --crop-extent-m 100 \
  --output runs/fulltoken_rgb
\`\`\`

For **strictly paired A4/A5** the GAT must be trained and selected on the **same RGB subset** used for A5 (the command does that). Training/evaluating RGB only on observed samples or using teacher prefix greater than zero changes the test distribution, so never compare those raw numbers to the old no-RGB 2808-query table without paired-filter controls.

### One RTX Pro 6000 96 GB, batch 16

- BF16 enabled for learnable graph/matcher stages; frozen BERT and SigLIP2 cached.
- Top-K SigLIP2 only processes actually observed candidate crops with valid central pixels; no inferred negative evidence for unseen proposals.
- 28×28×(≤36) candidate-to-landmark geometry is on CPU at data loading; first epoch may include caching cost.
- Save \`full_localization_metrics.csv\`, \`.json\`, and per-variant best checkpoint. Val Seen R@8 selects checkpoints, Val Unseen is reported as final held-out only.
- Report timing, peak GPU GiB, *observed RGB candidate rate*, anchor match rate, and map-version SHA256 files.
- First run A0-A4 to diagnose retrieval. A5 may have ZERO positive coverage at \`t=0\`; the trainer will intentionally abort if not a single proposal has valid RGB. Fix by measuring coverage or controlled teacher-prefix diagnostic, **not by cropping unobserved targets**.

## Important limits

1. This PR contains a **complete offline target-localization A0–A5 experimental pipeline**, NOT a finished simulator-integrated navigation/trajectory/stop model.
2. Existing GraphNav trajectory and Stop heads belong to a different network. Connecting them to this new candidate scorer requires an explicit new closed-loop adapter and separate tests.
3. Current fuzzy matching is a conservative option; exact equivalence to SBF's local name binding must be checked against the local SBF implementation.
4. Pure 2-D CityNav coordinates require correct XY/east/north axis validation relative to the original source map metadata.
5. No claim of improved Unseen recall or faster real GPU training is made without running Revised CityNav.
