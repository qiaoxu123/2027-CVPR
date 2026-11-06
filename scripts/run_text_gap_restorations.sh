#!/bin/bash
set -e
cd /home/20261024_1/Workspace/2027-CVPR-text-gap
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
P=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/relation_bearing_s42_20261106/parser_env/bin/python
ROOT=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs
COMMON=(--previous "$ROOT/structured_target_topk_20261106/full" --catalog /home/20261024_1/Workspace/DATA/sbfnav-data/cityrefer_original/objects.json)
"$P" -u -m scripts.diagnose_text_gap_restorations "${COMMON[@]}" --output "$ROOT/text_gap_diagnosis_20261106/restore_smoke_v2" --limit 2
"$P" -u -m scripts.diagnose_text_gap_restorations "${COMMON[@]}" --output "$ROOT/text_gap_diagnosis_20261106/restore_v2"
