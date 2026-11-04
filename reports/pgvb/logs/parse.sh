#!/bin/bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR/Experiments/GeomBeliefLandmarkChain
/home/20261024_1/Workspace/2027-CVPR/Experiments/instruction_audit_env/bin/python -u -m scripts.parse_train_instructions_structured --annotations /home/20261024_1/Workspace/DATA/sbfnav-data/original_citynav/citynav_train_seen.json --catalog /home/20261024_1/Workspace/DATA/sbfnav-data/cityrefer_original/objects.json --output /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/pgvb_structured_v1/language > /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/pgvb_structured_v1/parse.log 2>&1
