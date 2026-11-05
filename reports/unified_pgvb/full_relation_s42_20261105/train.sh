#!/bin/bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR/Experiments/GeomBeliefLandmarkChain
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
extra=()
if [ -f /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_full_relation_s42_20261105/core/last.pt ]; then extra+=(--resume); fi
/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python -u -m scripts.train_unified_full_relation --train-jsonl /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_full_relation_s42_20261105/train_revised.jsonl --stage1 /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/graphnav_v1_stage1/best.pt --original-v2 /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_pgvb_v2_pilot_s42/full_training/last.pt --output /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_full_relation_s42_20261105/core --epochs 5 --batch-size 8 --seed 42 "${extra[@]}" >> /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_full_relation_s42_20261105/train.log 2>&1
