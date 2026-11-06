#!/bin/bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR-relation-bearing
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
RUN=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/relation_bearing_s42_20261106
while ! grep -q '^Done relations' "$RUN/smoke_relations.log" 2>/dev/null; do
 if ! tmux has-session -t relation_bearing_smoke 2>/dev/null; then echo 'Smoke not verified complete';exit 1;fi
 sleep 5
done
/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python -u -m scripts.train_role_polygon_bearing --run "$RUN" --arm "$1" --epochs 12 --batch-size 16 --lr 0.0003 > "$RUN/train_$1.log" 2>&1
