#!/bin/bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR/Experiments/GeomBeliefLandmarkChain
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
task_root=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/pgvb_structured_v1
task_python=/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python
"$task_python" -u -m scripts.train_pgvb --data "$task_root/data" --output "$task_root/smoke_s0" --smoke-steps 2 --raw-backbone /home/20261024_1/.cache/huggingface/hub/models--google--siglip2-large-patch16-384/snapshots/1b426889ea62b5a72bf9839009a1b184bfc9c178 > "$task_root/smoke.log" 2>&1
"$task_python" -u -m scripts.train_pgvb --data "$task_root/data" --output "$task_root/smoke_s0" --smoke-steps 4 --resume > "$task_root/resume.log" 2>&1
"$task_python" -u -m scripts.train_pgvb --data "$task_root/data" --output "$task_root/pilot_s0" > "$task_root/train.log" 2>&1
printf '0\n' > "$task_root/train.exit"
