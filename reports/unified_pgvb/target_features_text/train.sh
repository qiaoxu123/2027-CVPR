#!/bin/bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR/Experiments/GeomBeliefLandmarkChain
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python
RUN=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_v2_target_features_s42_20261106
STAGE1=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/graphnav_v1_stage1/best.pt
V2=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_pgvb_v2_pilot_s42/full_training/last.pt
"$PY" -u -m scripts.train_unified_full_relation --train-jsonl "$RUN/inputs/target_features_landmark_relation.jsonl" --stage1 "$STAGE1" --original-v2 "$V2" --output "$RUN/smoke" --epochs 1 --batch-size 8 --seed 42 --max-batches 3 > "$RUN/smoke.log" 2>&1
"$PY" -u -m scripts.train_unified_full_relation --train-jsonl "$RUN/inputs/target_features_landmark_relation.jsonl" --stage1 "$STAGE1" --original-v2 "$V2" --output "$RUN/train" --epochs 5 --batch-size 8 --seed 42 > "$RUN/training.log" 2>&1
