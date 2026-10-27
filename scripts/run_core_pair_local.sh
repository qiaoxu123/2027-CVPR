#!/usr/bin/env bash
# Server-local reproducible entrypoint. All large outputs stay in ignored runs/.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
PYTHON=/home/20261024_1/miniconda3/envs/graphnav/bin/python
PROJECT=/home/20261024_1/Workspace/2027-CVPR
SBF_SOURCE="$PROJECT/Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source"
OUTPUT=runs/joint_rgb_geometry_core_20261027
mkdir -p "$OUTPUT"
exec 9>"$OUTPUT/execution.lock"
flock -n 9 || { echo 'Core pair already running'; exit 1; }
COMMON=(--cache "$PROJECT/GraphNav/data/rebuilt_stage1_20261025/frames" --sources reports/core_pair_20261027/initial_sources_local.json --bootstrap-checkpoint "$PROJECT/GraphNav/runs/graphnav_v1_stage1/initial_complete.pt" --bert-cache "$PROJECT/GraphNav/data/rebuilt_stage1_20261025/bert_tokens.pt" --sbf-source "$SBF_SOURCE" --sbf-config "$SBF_SOURCE/configs/sbfnav/main.yaml" --data-root /home/20261024_1/Workspace/DATA/sbfnav-data --batch-size 16 --workers 4 --seed 0)
"$PYTHON" -m pytest -q > reports/core_pair_20261027/pytest_full.txt 2>&1
"$PYTHON" -m compileall -q graphnav scripts
"$PYTHON" -c 'from torch.utils.tensorboard import SummaryWriter; import torch; assert torch.cuda.is_available()'
for arm in A0 A1; do
  if [ "$arm" = A0 ]; then fusion=original; else fusion=joint-rgb-geometry; fi
  for stage in A B C; do
    destination="$OUTPUT/smoke_${arm}_${stage}"
    if [ ! -f "$destination/complete.json" ]; then
      if [ -d "$destination" ]; then mv "$destination" "${destination}_failed_$(date +%s)"; fi
      "$PYTHON" -u -m scripts.train_graphnav_stage1 "${COMMON[@]}" --stage "$stage" --fusion-kind "$fusion" --epochs 1 --smoke --max-batches 2 --output "$destination" > "${destination}.log" 2>&1
    fi
  done
done
for arm in A0 A1; do
  if [ "$arm" = A0 ]; then model=graphnav-v1; else model=graphnav-joint-rgb; fi
  destination="$OUTPUT/smoke_${arm}_closedloop"
  if [ ! -f "$destination/metrics.json" ]; then
    if [ -d "$destination" ]; then mv "$destination" "${destination}_failed_$(date +%s)"; fi
    "$PYTHON" -u -m scripts.evaluate_closedloop --model "$model" --sbf-source "$SBF_SOURCE" --config "$SBF_SOURCE/configs/sbfnav/main.yaml" --data-root /home/20261024_1/Workspace/DATA/sbfnav-data --checkpoint "$OUTPUT/smoke_${arm}_C/best.pt" --split val_seen --max-episodes 2 --output "$destination" > "${destination}.log" 2>&1
  fi
done
if [ "${1:-}" = --smoke-only ]; then exit 0; fi
"$PYTHON" -u -m scripts.run_joint_rgb_geometry_core "${COMMON[@]}" --output "$OUTPUT"
