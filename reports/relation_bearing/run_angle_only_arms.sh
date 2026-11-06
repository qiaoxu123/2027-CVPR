#!/bin/bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR-relation-bearing
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
RUN=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/relation_bearing_s42_20261106
/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python -u -m scripts.train_role_polygon_angles_only --source-run "$RUN" --arm "$1" --epochs 12 --batch-size 16 --lr 0.0003 > "$RUN/train_angle_$1.log" 2>&1
