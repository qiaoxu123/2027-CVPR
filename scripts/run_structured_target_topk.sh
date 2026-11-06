#!/bin/bash
set -e
cd /home/20261024_1/Workspace/2027-CVPR-target-topk
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
P=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/relation_bearing_s42_20261106/parser_env/bin/python
ROOT=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs
COMMON=(--source "$ROOT/unified_v2_text_ablation_s42_20261105/paired_inputs/full_text.jsonl" --catalog /home/20261024_1/Workspace/DATA/sbfnav-data/cityrefer_original/objects.json --checkpoint "$ROOT/unified_v2_target_features_s42_20261106/train/last.pt")
"$P" -u -m scripts.diagnose_structured_target_topk "${COMMON[@]}" --output "$ROOT/structured_target_topk_20261106/smoke" --limit 2
"$P" -u -m scripts.diagnose_structured_target_topk "${COMMON[@]}" --output "$ROOT/structured_target_topk_20261106/full"
