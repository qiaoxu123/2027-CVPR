#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
PYTHON=/home/20261024_1/miniconda3/envs/graphnav/bin/python
PROJECT=/home/20261024_1/Workspace/2027-CVPR
SBF_SOURCE="$PROJECT/Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source"
OUTPUT=runs/joint_rgb_geometry_A1_20261027
mkdir -p "$OUTPUT"
exec 9>"$OUTPUT/execution.lock"
flock -n 9 || { echo 'A1 already running'; exit 1; }
"$PYTHON" -u -m scripts.run_joint_rgb_geometry_core --a1-only --a0-reference-root "$PROJECT/GraphNav/runs/graphnav_v1_stage1" --cache "$PROJECT/GraphNav/data/rebuilt_stage1_20261025/frames" --sources reports/core_pair_20261027/initial_sources_local.json --bootstrap-checkpoint "$PROJECT/GraphNav/runs/graphnav_v1_stage1/initial_complete.pt" --bert-cache "$PROJECT/GraphNav/data/rebuilt_stage1_20261025/bert_tokens.pt" --sbf-source "$SBF_SOURCE" --sbf-config "$SBF_SOURCE/configs/sbfnav/main.yaml" --data-root /home/20261024_1/Workspace/DATA/sbfnav-data --batch-size 16 --workers 4 --seed 0 --output "$OUTPUT"
