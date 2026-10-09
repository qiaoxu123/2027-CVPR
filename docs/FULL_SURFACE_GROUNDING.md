# Typed Surface Landmark Grounding (2026-10-09)

This branch starts from **validation/static-geometry-20261009** and retains its 38-test baseline/report unmodified. It **changes the static input distribution**, so the previously reported 6.13% relation grounding / 16.17% Unseen R@8 must **not** be presented as new results.

## Root cause addressed

The former static-only converter dropped all referenced entities except `object_type=Building`. CityRefer contains exact-world-coordinate **area polygons** with types `TrafficRoad`, `Footpath`, `Bridge`, `Parking`, `Ground`, and other entities. These are **not road centerlines** and carry **no verified OSM road connectivity**.

Expanded model preserves **one entire area polygon per uniquely identified CityRefer object**. Each named polygon retains its `id`, `name`, `object_type`, `contour`, and matched `aliases`. Typed polygon nodes, along/near/between/east geometry, and GAT edges share the *same static visible input*. Per-polygon vertices are used only as shape features.

### Name binding

Canonical punctuation/case normalization plus conservative source-type-specific suffix mapping:
- “BMTR building” → mapped **Building** called “BMTR”
- “One Stop Building” → mapped **Building** called “One Stop”
- “Broadway Road” → mapped **TrafficRoad** called “Broadway”
- No generic fuzzy matching, no “closest to GT”, no forced choice between homonymous source IDs.
- Explicit named anchor relation parsing accepts aliases, “to the east of X”, “near X”, “between X and Y”, and “along X”; parser remains conservative and does not claim full language coverage.

### Exact comparators

The new CSV writes an extra **building_prior** to measure the recovery of nonbuilding surfaces on exactly the same queries and 28×28 grid:

| Variant | Source |
|---|---|
| building_prior | Old building-only landmark prior implementation on NEW samples |
| prior (S0) | All named typed CityRefer polygons |
| rules (S1) | All polygons + grounded explicit spatial constraints |
| gat (S2) | S1 + typed Edge-GATv2 trained from scratch |

All R@1, R@4, R@8 /20m and Top-1 NE use identical query denominators. Choose checkpoint using Val Seen R@8 only; Val Unseen is held out. The historic S0 was not the same as the old HETT reference-mask prior, so do not claim bitwise parity.

### Prepare fresh CityNav data (mandatory)

**Do not reuse stale `data/graphnav_*.jsonl` or old `*.raw.jsonl`:** the previous data was building-only. After making the new static JSONL, explicitly refresh the deduplication backups:

```bash
git fetch origin
git checkout 2027-CVPR/full-landmark-geometry-grounding
pip install -c configs/static_geometry_rtx6000_constraints.txt -e ".[dev,foundation]"

# Reuse the same objects/processed/trajectories/world-bounds paths documented
# in the previous validation report. Repeat conversion for each split:
python -m scripts.prepare_citynav \
  --objects "$CITYNAV_SOURCE/cityrefer/objects.json" \
  --processed "$CITYNAV_SOURCE/cityrefer/processed_descriptions.json" \
  --trajectories "$CITYNAV_SOURCE/citynav/citynav_train_seen.json" \
  --bounds data/citynav_bounds.json \
  --static-only \
  --output data/graphnav_train_seen.jsonl

# Repeat for val_seen and val_unseen, then:
python -m scripts.prepare_static_splits \
  --data-dir data --refresh-raw \
  --output runs/expanded_static/query_deduplication.json

python -m scripts.audit_static_data \
  --data-dir data --source-dir "$CITYNAV_SOURCE" \
  --output runs/expanded_static/data_audit.json
```

Use `CITYNAV_SOURCE=/home/20260922_1/Workspace/DATA/citynav/extracted/data` only if that directory is still present on the local machine.

Inspect **source-type counts**, percent of queries with any surface anchor, relation-grounded fraction, and aliases resolved BEFORE training. Keep the same deterministic split cleaning: unique map+instruction, val seen/unseen unchanged, 14 duplicate train-vs-val queries removed if still present. Check no GT coordinates or future RGB entered the model.

### Train first: geometry only

Hardware: **single RTX Pro 6000 96GB, BF16, batch size 16**.

```bash
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python -m graphnav.static_experiment \
  --train data/graphnav_train_seen.jsonl \
  --val-seen data/graphnav_val_seen.jsonl \
  --val-unseen data/graphnav_val_unseen.jsonl \
  --device cuda --batch-size 16 --epochs 5 --max-roads 0 \
  --output runs/expanded_static_bs16
```

`--max-roads 0` means no **verified polyline graph**. TrafficRoad/Footpath/Bridge **area polygons remain present via `landmarks`**. Never synthesize network topology from an area polygon. RGB/Depth, Waypoints, Stop, closed-loop SR/SPL remain excluded.

Report R@1/R@4/R@8/NE on Val Seen + Unseen for the four variants and two extra diagnostic subsets: **any named surface**, **no grounded surface**. Compare paired gains (S0 vs building_prior, S1 vs S0, S2 vs S1) and per-type target groups Car / Building / Parking / Ground. Only if Unseen improves should observed-only SigLIP2 and trajectory navigation be introduced.

### Code

- `graphnav/landmark_alias.py`: fast indexed, source-type-safe aliases preserving source object identity
- `scripts/prepare_citynav.py`: all source polygons in static JSONL
- `graphnav/semantic_geometry.py`: typed closed region polygon vs verified open road polyline
- `graphnav/semantic_graph.py`: semantic relation nodes, type-conditioned GAT edges
- `graphnav/static_localization.py`: all-surface goal prior + building-only comparison
- `scripts/audit_static_data.py`: full-type, alias, and source contour roundtrip audit
- `scripts/prepare_static_splits.py`: explicit `--refresh-raw` for new conversion
- `tests/test_full_landmark_grounding.py`: regression tests for alias, region, no road topology and GT isolation

**No empirical R@K improvements are claimed until the local CityNav validation is rerun.**
