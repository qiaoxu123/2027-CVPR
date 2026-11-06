#!/bin/bash
set -e
cd /home/20261024_1/Workspace/2027-CVPR-text-gap
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
P=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/relation_bearing_s42_20261106/parser_env/bin/python
ROOT=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs
"$P" -u -m scripts.diagnose_text_gap_factors --previous "$ROOT/structured_target_topk_20261106/full" --output "$ROOT/text_gap_diagnosis_20261106/factor_smoke" --limit 2
"$P" -u -m scripts.diagnose_text_gap_factors --previous "$ROOT/structured_target_topk_20261106/full" --output "$ROOT/text_gap_diagnosis_20261106/factors"
