#!/bin/bash
set -e
cd /home/20261024_1/Workspace/2027-CVPR-text-gap
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/relation_bearing_s42_20261106/parser_env/bin/python -u -m scripts.diagnose_text_gap_words --previous /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/structured_target_topk_20261106/full --output /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/text_gap_diagnosis_20261106/words
