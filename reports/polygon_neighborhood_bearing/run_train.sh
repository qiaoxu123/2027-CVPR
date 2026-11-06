#!/bin/bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR-neighborhood-bearing
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
PY=/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python
BASE=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs
RUN=$BASE/polygon_neighborhood_bearing_s42_20261106
mkdir -p "$RUN"
COMMON=(--train-jsonl "$BASE/unified_full_relation_s42_20261105/train_revised.jsonl" --existing-full-checkpoint "$BASE/unified_v2_target_features_s42_20261106/train/last.pt" --public-objects /home/20261024_1/Workspace/DATA/sbfnav-data/cityrefer_original/objects.json --seed 42 --batch-size 32 --lr 1e-4 --device cuda)
"$PY" -u -m scripts.train_polygon_neighborhood_bearing "${COMMON[@]}" --output "$RUN/smoke" --epochs 1 --max-train-batches 3 --max-eval-batches 2 > "$RUN/smoke.log" 2>&1
"$PY" -u -m scripts.train_polygon_neighborhood_bearing "${COMMON[@]}" --output "$RUN/full" --epochs 3 > "$RUN/full.log" 2>&1
