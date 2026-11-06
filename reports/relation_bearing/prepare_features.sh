#!/bin/bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR-relation-bearing
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
RUN=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/relation_bearing_s42_20261106
while [ ! -f "$RUN/data/parsed.jsonl" ]; do sleep 5; done
/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python -u -m scripts.prepare_relation_bearing_features --run "$RUN" --data-manifest /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/polygon_neighborhood_bearing_s42_20261106/full/data_manifest.json --public-objects /home/20261024_1/Workspace/DATA/sbfnav-data/cityrefer_original/objects.json --bert-cache /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/polygon_neighborhood_bearing_s42_20261106/full/bert_cache.pt > "$RUN/features.log" 2>&1
