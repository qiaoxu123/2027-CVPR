#!/bin/bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR/Experiments/GeomBeliefLandmarkChain
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python
BASE=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs
RUN=$BASE/unified_angle_only_s42_20261106
OBJECTS=/home/20261024_1/Workspace/DATA/sbfnav-data/cityrefer_original/objects.json
"$PY" -u -m scripts.train_unified_angle_only --train-jsonl "$BASE/unified_full_relation_s42_20261105/train_revised.jsonl" --existing-full-checkpoint "$BASE/unified_full_relation_s42_20261105/core/last.pt" --public-objects "$OBJECTS" --output "$RUN/smoke" --epochs 1 --batch-size 8 --seed 42 --max-batches 12 > "$RUN/smoke.log" 2>&1
"$PY" -u -m scripts.train_unified_angle_only --train-jsonl "$BASE/unified_full_relation_s42_20261105/train_revised.jsonl" --existing-full-checkpoint "$BASE/unified_full_relation_s42_20261105/core/last.pt" --public-objects "$OBJECTS" --output "$RUN/full" --epochs 3 --batch-size 8 --seed 42 > "$RUN/training.log" 2>&1
