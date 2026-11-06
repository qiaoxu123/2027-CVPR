#!/usr/bin/env bash
set -euo pipefail
# Run from this PR's repository root. Output must not already exist.
PY=/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python
BASE=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs
DATA=/home/20261024_1/Workspace/DATA/sbfnav-data
"$PY" -m pytest -q tests/test_val_seen_binding_failures.py tests/test_unified_polygon_group.py tests/test_unified_text_rule_controls.py
"$PY" -m scripts.report_val_seen_binding_failures \
  --paired reports/unified_pgvb/angle_only/paired2470.csv \
  --episodes reports/unified_pgvb/angle_only/episodes2470.json \
  --baseline-root "$BASE/unified_v2_text_ablation_s42_20261105/eval_full_text" \
  --comparison-root "$BASE/unified_angle_only_s42_20261106/val_seen_predicted_angles" \
  --public-objects "$DATA/cityrefer_original/objects.json" \
  --annotations "$DATA/processed_citynav/citynav_val_seen.json" \
  --output "${1:-/tmp/val_seen_binding_report_reproduced}"
"$PY" -m scripts.verify_val_seen_binding_report "${1:-/tmp/val_seen_binding_report_reproduced}"
