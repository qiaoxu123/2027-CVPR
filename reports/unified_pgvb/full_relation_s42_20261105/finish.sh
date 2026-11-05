#!/bin/bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR/Experiments/GeomBeliefLandmarkChain
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python
RUN=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_full_relation_s42_20261105
STAGE=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/graphnav_v1_stage1/best.pt
V2=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_pgvb_v2_pilot_s42/full_training/last.pt
SBF=/home/20261024_1/Workspace/2027-CVPR/Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source
DATA=/home/20261024_1/Workspace/DATA/sbfnav-data
while [ ! -f "$RUN/train.exit" ]; do sleep 10; done
if [ "$(cat "$RUN/train.exit")" != 0 ]; then echo 'Core training failed; pipeline stopped'; exit 1; fi
"$PY" -c 'import json,sys;assert json.load(open(sys.argv[1]))["complete"]' "$RUN/core/progress.json"
extra=()
if [ -f "$RUN/visual/last.pt" ]; then extra+=(--resume); fi
"$PY" -u -m scripts.train_unified_full_visual --base-checkpoint "$RUN/core/last.pt" --inventory /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_direct_target_s42_20261105/inventory_validated --cache /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_direct_target_s42_20261105/cache_aligned --output "$RUN/visual" --epochs 3 --batch-size 4 --seed 42 "${extra[@]}" > "$RUN/visual.log" 2>&1
"$PY" -m scripts.smoke_full_relation_runtime --stage1 "$STAGE" --original-v2 "$V2" --checkpoint "$RUN/visual/last.pt" --data-root "$DATA" --sbf-source "$SBF" --config "$SBF/configs/sbfnav/main.yaml" --output "$RUN/runtime_smoke" > "$RUN/runtime_smoke.log" 2>&1
for arm in original_geometry geometry relation rgb; do
  checkpoint="$RUN/core/last.pt"
  relation=off
  visual=off
  geometry=trained
  if [ "$arm" = original_geometry ]; then geometry=original; fi
  if [ "$arm" = relation ] || [ "$arm" = rgb ]; then relation=full; fi
  if [ "$arm" = rgb ]; then visual=tracked; checkpoint="$RUN/visual/last.pt"; fi
  extra=()
  if [ -d "$RUN/eval_$arm" ]; then extra+=(--resume); fi
  "$PY" -u -m scripts.evaluate_closedloop --model unified-relation --split val_seen --sbf-source "$SBF" --config "$SBF/configs/sbfnav/main.yaml" --data-root "$DATA" --checkpoint "$STAGE" --unified-v2 "$V2" --relation-checkpoint "$checkpoint" --relation-control "$relation" --relation-visual-control "$visual" --relation-geometry-source "$geometry" --output "$RUN/eval_$arm" "${extra[@]}" > "$RUN/eval_$arm.log" 2>&1
 done
