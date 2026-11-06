#!/bin/bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR-v3
export PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
R=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_v3_full_s42_20261106
PY=/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python
while tmux has-session -t '=unified_v3_full_train' 2>/dev/null; do sleep 5; done
"$PY" -c 'import json,sys;d=json.load(open(sys.argv[1]))["manifest"];assert d["epochs"]>0 and d["stop_reason"] in ("train_loss_plateau","time_budget","max_epochs")' "$R/train/report.json"
"$PY" -u -m scripts.evaluate_unified_pgvb_v3_initial --sbf-source /home/20261024_1/Workspace/2027-CVPR/Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source --data-root /home/20261024_1/Workspace/DATA/sbfnav-data --objects /home/20261024_1/Workspace/DATA/sbfnav-data/cityrefer_original/objects.json --source-v2 /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_pgvb_v2_pilot_s42/full_training/last.pt --checkpoint "$R/train/last.pt" --output "$R/val_seen_initial_full"

if "$PY" -c 'import json,sys;d=json.load(open(sys.argv[1]));sys.exit(0 if d["navigation_gate_passed"] else 1)' "$R/val_seen_initial_full/summary.json"; then
  for MODE in geometry fused; do
    EXTRA=()
    if [ "$MODE" = geometry ]; then EXTRA=(--geometry-only); fi
    "$PY" -u -m scripts.evaluate_unified_pgvb_v3_navigation --sbf-source /home/20261024_1/Workspace/2027-CVPR/Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source --config /home/20261024_1/Workspace/2027-CVPR/Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source/configs/sbfnav/main.yaml --data-root /home/20261024_1/Workspace/DATA/sbfnav-data --source-v2 /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_pgvb_v2_pilot_s42/full_training/last.pt --checkpoint "$R/train/last.pt" --stage1 /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/graphnav_v1_stage1/best.pt --initial-eval "$R/val_seen_initial_full" --output "$R/navigation_$MODE" "${EXTRA[@]}"
  done
else
  echo "Full initial Top1 below 55% or errors: navigation not started; TRAIN diagnosis required."
fi
