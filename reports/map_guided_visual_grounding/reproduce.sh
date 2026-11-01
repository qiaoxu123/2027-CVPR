#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN=${PYTHON_BIN:-/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/home/20261024_1/Workspace/2027-CVPR/Ablations/GraphNav/visual-residual-20261013}
DATA_ROOT=${DATA_ROOT:-/home/20261024_1/Workspace/DATA/sbfnav-data}
OUTPUT_ROOT=${OUTPUT_ROOT:-runs/map_guided_reproduce}
common=(--native-checkpoint "$EXPERIMENT_ROOT/aerialvg_native_plain_sigma8_s0_20261101/best.pt"
        --catalog "$DATA_ROOT/cityrefer/objects.json"
        --train-annotations "$DATA_ROOT/processed_citynav/citynav_train_seen.json")
"$PYTHON_BIN" -m pytest -q tests/test_map_guided_grounding.py tests/test_map_guided_analysis.py tests/test_observed_landmarks.py
"$PYTHON_BIN" -m scripts.diagnose_map_guided_visual_grounding "${common[@]}" --cache "$EXPERIMENT_ROOT/aerialvg_native_training_train256_20261101" --partition train --limit 48 --output "$OUTPUT_ROOT/train_fit48"
"$PYTHON_BIN" -m scripts.diagnose_map_guided_visual_grounding "${common[@]}" --cache "$EXPERIMENT_ROOT/aerialvg_native_training_train256_20261101" --partition train --cohort calibration --freeze-protocol "$OUTPUT_ROOT/train_fit48/protocol.json" --output "$OUTPUT_ROOT/train_calibration248"
"$PYTHON_BIN" -m scripts.diagnose_map_guided_visual_grounding "${common[@]}" --cache "$EXPERIMENT_ROOT/aerialvg_native_training_dev128_20261101" --partition dev --freeze-protocol "$OUTPUT_ROOT/train_fit48/protocol.json" --output "$OUTPUT_ROOT/dev635"
"$PYTHON_BIN" -m scripts.analyze_map_guided_diagnostic --records "$OUTPUT_ROOT/dev635/records.jsonl" --cache "$EXPERIMENT_ROOT/aerialvg_native_training_dev128_20261101" --catalog "$DATA_ROOT/cityrefer/objects.json" --train-annotations "$DATA_ROOT/processed_citynav/citynav_train_seen.json" --output "$OUTPUT_ROOT/dev635/analysis.json"
"$PYTHON_BIN" -m scripts.audit_map_guided_instance_proposals --records "$OUTPUT_ROOT/dev635/records.jsonl" --catalog "$DATA_ROOT/cityrefer/objects.json" --output "$OUTPUT_ROOT/dev635/instance_proposal_oracle.json"
"$PYTHON_BIN" -m scripts.audit_map_guided_strict_shuffle --records "$OUTPUT_ROOT/dev635/records.jsonl" --cache "$EXPERIMENT_ROOT/aerialvg_native_training_dev128_20261101" --catalog "$DATA_ROOT/cityrefer/objects.json" --reference-cache "$OUTPUT_ROOT/dev635/reference_features" --native-checkpoint "$EXPERIMENT_ROOT/aerialvg_native_plain_sigma8_s0_20261101/best.pt" --output "$OUTPUT_ROOT/strict_shuffle_audit"
"$PYTHON_BIN" -m scripts.profile_map_guided_observed_rois --train-records "$OUTPUT_ROOT/train_fit48/records.jsonl" --catalog "$DATA_ROOT/cityrefer/objects.json" --output "$OUTPUT_ROOT/train_fit48/online_profile.json"
# After a disconnect, rerun the interrupted diagnose command with --resume.
# Never regenerate pixels/regions from unvisited poses or modify frozen constants.
