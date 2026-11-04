#!/bin/bash
set -uo pipefail
cd /home/20261024_1/Workspace/2027-CVPR/Experiments/GeomBeliefLandmarkChain
/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python -u -m scripts.train_direction_belief \
 --data /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/direction_belief_v1/data_split \
 --output /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/direction_belief_v1/pilot_s0 \
 > /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/direction_belief_v1/execution/train.log 2>&1
printf '%s\n' "$?" > /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/direction_belief_v1/execution/train.exit
