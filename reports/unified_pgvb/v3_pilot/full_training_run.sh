#!/bin/bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR-v3
export PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python
R=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_v3_full_s42_20261106
"$PY" -u -m scripts.unified_pgvb_v3_pilot cache --source-v2 /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_pgvb_v2_pilot_s42/full_training/last.pt --train-jsonl /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_full_relation_s42_20261105/train_revised.jsonl --objects /home/20261024_1/Workspace/DATA/sbfnav-data/cityrefer_original/objects.json --ground-levels /home/20261024_1/Workspace/2027-CVPR-Under-Test/configs/ground_levels.json --rgbd-dir /home/20261024_1/Workspace/DATA/sbfnav-data/rgbd --cache "$R/cache" --limit 0 --views 5 --fixed-yaw 0 --altitude-m 80 --offset-m 35 --device cuda

"$PY" -u -m scripts.unified_pgvb_v3_pilot train --source-v2 /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_pgvb_v2_pilot_s42/full_training/last.pt --initialize-v3 /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_v3_pilot_s42_20261106/train1000/last.pt --cache "$R/cache" --output "$R/train" --epochs 100 --batch-size 1 --seed 42 --max-training-hours 4 --min-epochs 3 --plateau-patience 2 --plateau-relative 0.005 --checkpoint-updates 1000 --skip-evaluation --device cuda
