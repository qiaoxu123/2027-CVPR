#!/bin/sh
set -eu
cd /home/rental/20260922_1/Workspace/2027-CVPR/Ablations/GraphNav/visual-residual-20261013
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
PYTHON=/home/rental/20260922_1/Workspace/2027-CVPR/Baselines/.reproduction-env/bin/python
"$PYTHON" -m scripts.run_polygon_ablation --resume
"$PYTHON" -m scripts.polygon_ablation_statistics
"$PYTHON" -m scripts.sync_polygon_ablation
