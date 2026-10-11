# P3 measured-point multi-view diagnosis

See `docs/P3_MULTIVIEW_RGB_GROUNDING_REPORT.md`. No training or Goal updates.
All source DATA and prior checkpoints are read-only. Use the existing remoteclip
venv. Run from the repository root, with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.
Use `HF_HUB_OFFLINE=1` for cached model inference. External resources use direct
HTTP first and environment proxy only after failure.

Executed render commands:

```bash
python -m experiments.multiview_rgb_grounding.render --plan runs/instance_grounding/plan_train --output runs/multiview_rgb_grounding/render_pilot_train_scatter --limit 6 --prefix-policy earliest
python -m experiments.multiview_rgb_grounding.render --plan runs/instance_grounding/plan_dev --output runs/multiview_rgb_grounding/render_pilot_dev_scatter --limit 8 --prefix-policy latest
python -m experiments.multiview_rgb_grounding.dev_probe
python -m experiments.multiview_rgb_grounding.render --plan runs/multiview_rgb_grounding/plan_dev_error_probe --output runs/multiview_rgb_grounding/render_dev_error_probe --radii 1
python -m experiments.multiview_rgb_grounding.quality --pilots render_dev_error_probe --output runs/multiview_rgb_grounding/quality_dev_error_probe
```

Output directories are exclusive. Do not overwrite historical or partial results.
The original first attempts and initial grid-unit error outputs are retained in
ignored runs; authoritative matching output ends in `_world` for Train and
`matching_dev_error_probe` for Dev. Grid coordinates must be transformed by
`geometry.world_grid`, which matches the frozen P2.5 baseline.

Full frame RGB/depth/visibility are sensor-simulator artifacts, not existing
CityFlight oblique captures. Only current measured-point crops go to encoders.
Instance assignment is a weak Polygon/3D-box proxy; ambiguity is Unknown.
GT labels never aim cameras, select crops, or enter model text/image inputs.

`deliver` packages compact diagnostics. `geometry_baseline` adds the instance
baseline from exactly the same frozen candidate IDs. These are not Goal-grid
reranking metrics. Assets contain compressed predictions and aggregate audits;
weights, point clouds, feature tensors and external raw downloads stay local.
