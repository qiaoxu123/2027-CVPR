#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN=${PYTHON_BIN:-/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/home/20261024_1/Workspace/2027-CVPR/Ablations/GraphNav/visual-residual-20261013}
PUBLIC_CACHE=${PUBLIC_CACHE:-/home/20261024_1/Workspace/2027-CVPR/GraphNav/data/rebuilt_stage1_20261025/frames}
OLD_PILOT=${OLD_PILOT:-/home/20261024_1/Workspace/2027-CVPR/Ablations/GraphNav/aerialvg-relation-gated-validation-20261101/runs/relation_gate_full_s0/pilot.pt}
OUTPUT_ROOT=${OUTPUT_ROOT:-runs/native_preserving_reproduce_s0}
common=(--train "$EXPERIMENT_ROOT/aerialvg_native_training_train256_20261101"
        --dev "$EXPERIMENT_ROOT/aerialvg_native_training_dev128_20261101"
        --stage1-cache "$PUBLIC_CACHE"
        --native-checkpoint "$EXPERIMENT_ROOT/aerialvg_native_plain_sigma8_s0_20261101/best.pt"
        --old-pilot "$OLD_PILOT")
"$PYTHON_BIN" -m pytest -q tests/test_native_preserving_gate.py tests/test_native_multireference.py tests/test_native_relation_grounding.py tests/test_native_gated_belief_fusion.py
"$PYTHON_BIN" -m scripts.train_native_preserving_relation_gate "${common[@]}" --output "$OUTPUT_ROOT/smoke" --epochs 1 --gate-epochs 1 --seed 0
"$PYTHON_BIN" -m scripts.train_native_preserving_relation_gate "${common[@]}" --output "$OUTPUT_ROOT/pilot" --epochs 4 --gate-epochs 3 --seed 0 --evaluate-dev
"$PYTHON_BIN" -m scripts.audit_native_relation_training_support --train "$EXPERIMENT_ROOT/aerialvg_native_training_train256_20261101" --stage1-cache "$PUBLIC_CACHE" --output "$OUTPUT_ROOT/train_support_audit.json"
