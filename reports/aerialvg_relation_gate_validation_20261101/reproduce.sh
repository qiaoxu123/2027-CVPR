#!/usr/bin/env bash
set -euo pipefail
# Run at repository root; every output must be new to preserve prior runs.
PYTHON_BIN=${PYTHON_BIN:-/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/home/20261024_1/Workspace/2027-CVPR/Ablations/GraphNav/visual-residual-20261013}
PUBLIC_CACHE=${PUBLIC_CACHE:-/home/20261024_1/Workspace/2027-CVPR/GraphNav/data/rebuilt_stage1_20261025/frames}
OUTPUT_ROOT=${OUTPUT_ROOT:-runs/relation_gate_reproduce_s0}
common=(--train "$EXPERIMENT_ROOT/aerialvg_native_training_train256_20261101"
        --dev "$EXPERIMENT_ROOT/aerialvg_native_training_dev128_20261101"
        --stage1-cache "$PUBLIC_CACHE"
        --native-checkpoint "$EXPERIMENT_ROOT/aerialvg_native_plain_sigma8_s0_20261101/best.pt")
"$PYTHON_BIN" -m pytest -q tests/test_native_relation_grounding.py tests/test_native_gated_belief_fusion.py
"$PYTHON_BIN" -m scripts.train_native_relational_grounding "${common[@]}" --output "$OUTPUT_ROOT/smoke" --epochs 1 --gate-epochs 1 --seed 0
"$PYTHON_BIN" -m scripts.train_native_relational_grounding "${common[@]}" --output "$OUTPUT_ROOT/full" --epochs 4 --gate-epochs 3 --seed 0
"$PYTHON_BIN" -m scripts.diagnose_native_relational_grounding "${common[@]}" --pilot "$OUTPUT_ROOT/full/pilot.pt" --output "$OUTPUT_ROOT/full/diagnostic.json"
"$PYTHON_BIN" -m scripts.audit_native_public_relations --train "${common[1]}" --dev "${common[3]}" --stage1-cache "$PUBLIC_CACHE" --output "$OUTPUT_ROOT/public_relation_audit.json"
