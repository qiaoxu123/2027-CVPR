#!/bin/bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR/Experiments/GeomBeliefLandmarkChain
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
task_root=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/geometry_guided_visual_belief
task_python=/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python
common=(--data "$task_root/landmark_data_augmented_v1" --teacher "$task_root/teacher_v1" --catalog /home/20261024_1/Workspace/DATA/sbfnav-data/cityrefer_original/objects.json)
"$task_python" -u -m scripts.train_world_aligned_landmark_visual "${common[@]}" --output "$task_root/world_smoke_s0_v2" --smoke-steps 2 > /tmp/geometry_guided_visual_belief/world_smoke.log 2>&1
"$task_python" -u -m scripts.train_world_aligned_landmark_visual "${common[@]}" --output "$task_root/world_smoke_s0_v2" --smoke-steps 4 --resume > /tmp/geometry_guided_visual_belief/world_resume.log 2>&1
"$task_python" -u -m scripts.train_world_aligned_landmark_visual "${common[@]}" --output "$task_root/world_pilot_s0_v2" > /tmp/geometry_guided_visual_belief/world_train.log 2>&1
"$task_python" -u -m scripts.train_world_aligned_landmark_visual "${common[@]}" --output "$task_root/world_pilot_s0_v2" --evaluate > /tmp/geometry_guided_visual_belief/world_evaluate.log 2>&1
printf '0\n' > /tmp/geometry_guided_visual_belief/world_train.exit
