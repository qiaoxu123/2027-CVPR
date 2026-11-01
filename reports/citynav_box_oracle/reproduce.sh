#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN=${PYTHON_BIN:-/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/home/20261024_1/Workspace/2027-CVPR/Ablations/GraphNav/visual-residual-20261013}
DATA_ROOT=${DATA_ROOT:-/home/20261024_1/Workspace/DATA/sbfnav-data}
OUTPUT_ROOT=${OUTPUT_ROOT:-runs/citynav_box_oracle_reproduce}
CACHE="$EXPERIMENT_ROOT/aerialvg_native_training_train256_20261101"
NATIVE_CHECKPOINT="$EXPERIMENT_ROOT/aerialvg_native_plain_sigma8_s0_20261101/best.pt"
"$PYTHON_BIN" -m pytest -q tests/test_citynav_box_diagnostics.py tests/test_map_guided_grounding.py tests/test_native_gated_belief_fusion.py
"$PYTHON_BIN" -m scripts.prepare_citynav_box_annotations --cache "$CACHE" --catalog "$DATA_ROOT/cityrefer/objects.json" --annotations "$DATA_ROOT/processed_citynav/citynav_train_seen.json" --output "$OUTPUT_ROOT/annotation_set" --review-size 64
common=(--cache "$CACHE" --annotations "$OUTPUT_ROOT/annotation_set/annotations.jsonl" --native-checkpoint "$NATIVE_CHECKPOINT")
# NO human labels: this is ONLY a projection hypothesis, never verified pixel GT.
"$PYTHON_BIN" -m scripts.diagnose_citynav_box_oracles "${common[@]}" --reviews "$OUTPUT_ROOT/annotation_set/reviews.unreviewed.json" --oracle-mode projection-hypothesis --output "$OUTPUT_ROOT/projection_sensitivity"
"$PYTHON_BIN" -m scripts.analyze_citynav_oracle_failures --records "$OUTPUT_ROOT/projection_sensitivity/records.jsonl" --cache "$CACHE" --native-checkpoint "$NATIVE_CHECKPOINT" --output "$OUTPUT_ROOT/projection_sensitivity/attribution.json"
"$PYTHON_BIN" -m scripts.audit_citynav_fusion_algebra --records "$OUTPUT_ROOT/projection_sensitivity/records.jsonl" --cache "$CACHE" --native-checkpoint "$NATIVE_CHECKPOINT" --output "$OUTPUT_ROOT/projection_sensitivity/fusion_algebra.json"
"$PYTHON_BIN" -m scripts.summarize_citynav_box_quality --annotations "$OUTPUT_ROOT/annotation_set/annotations.jsonl" --reviews "$OUTPUT_ROOT/annotation_set/reviews.unreviewed.json" --records "$OUTPUT_ROOT/projection_sensitivity/records.jsonl" --output "$OUTPUT_ROOT/projection_sensitivity/box_quality_grouped.json"
# Actual human reviews, if later available: pass EXPORTED human JSON and use a NEW output.
# "$PYTHON_BIN" -m scripts.diagnose_citynav_box_oracles "${common[@]}" --reviews /path/to/citynav_human_reviews.json --oracle-mode verified --output "$OUTPUT_ROOT/human_verified_oracles"
# --oracle-mode normal never injects private boxes; --resume requires unchanged source/protocol.
