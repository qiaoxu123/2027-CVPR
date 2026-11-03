#!/bin/bash
set -e
trap 'printf "%s\n" "$?" > /tmp/native_grid_weak27/exit_code' EXIT
cd /home/20261024_1/Workspace/2027-CVPR/Experiments/GeomBeliefLandmarkChain
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
R=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/landmark_chain_visual
PY=/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python
$PY -m scripts.prepare_native_grid_pilot --root $R --annotations $R/target_identity_ai_review48_labels_v1/ai_weak_boxes_v2.json --catalog /home/20261024_1/Workspace/DATA/sbfnav-data/cityrefer/objects.json --static-queries /home/20261024_1/Workspace/2027-CVPR/GraphNav/data/rebuilt_stage1_20261025/frames/queries.jsonl --output $R/native_grid_weak27_selection_v1
$PY -m scripts.run_native_grid_pilot --selection $R/native_grid_weak27_selection_v1 --checkpoint $R/world_v3_envelope_s0/best.pt --backbone /home/20261024_1/.cache/huggingface/hub/models--google--siglip2-large-patch16-384/snapshots/1b426889ea62b5a72bf9839009a1b184bfc9c178 --output $R/native_grid_weak27_forward_v1
