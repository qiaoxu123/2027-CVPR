#!/usr/bin/env bash
set -euo pipefail
cd /home/rental/20260922_1/Workspace/2027-CVPR/Ablations/GraphNav/visual-residual-20261013
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
GEOMBELIEF_PY=/home/rental/20260922_1/Workspace/2027-CVPR/Baselines/.reproduction-env/bin/python
"$GEOMBELIEF_PY" -m pytest -q tests/test_geombelief_inputs.py tests/test_geombelief_control.py
"$GEOMBELIEF_PY" -m scripts.geombelief_map_experiments
"$GEOMBELIEF_PY" -m scripts.geombelief_control_experiments
"$GEOMBELIEF_PY" -m scripts.geombelief_results
"$GEOMBELIEF_PY" -m scripts.geombelief_cases
