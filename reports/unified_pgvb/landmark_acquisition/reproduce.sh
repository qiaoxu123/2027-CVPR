#!/bin/bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR/Experiments/GeomBeliefLandmarkChain
PY=/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python
RUN=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/landmark_acquisition_20261105
ARGS=(--selection /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/named_landmark_navigation/selection64_v1 --checkpoint /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/graphnav_v1_stage1/best.pt --unified-checkpoint /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_pgvb_v2_pilot_s42/full_training/last.pt --sbf-source /home/20261024_1/Workspace/2027-CVPR/Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source --config /home/20261024_1/Workspace/2027-CVPR/Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source/configs/sbfnav/main.yaml --data-root /home/20261024_1/Workspace/DATA/sbfnav-data --methods Geometry LegacyPGVB TrackedPGVB)
$PY -m scripts.run_unified_pgvb_landmark_closedloop "${ARGS[@]}" --max-new-tasks 2 --output "$RUN/smoke" > "$RUN/smoke.log" 2>&1
$PY -c 'import json,sys; r=[json.loads(x) for x in open(sys.argv[1])]; assert len(r)==6; assert all(len(x["step_diagnostics"])>0 for x in r); print("Six paired smoke rollouts complete")' "$RUN/smoke/predictions.jsonl"
$PY -m scripts.run_unified_pgvb_landmark_closedloop "${ARGS[@]}" --output "$RUN/full64" > "$RUN/full64.log" 2>&1
