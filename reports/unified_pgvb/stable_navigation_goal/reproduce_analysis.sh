#!/bin/bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR/Experiments/GeomBeliefLandmarkChain
PY=/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python
RUN=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/stable_landmark_20261105/full64
SEL=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/named_landmark_navigation/selection64_v1
OUT=reports/unified_pgvb/stable_navigation_goal
$PY -m scripts.analyze_landmark_acquisition --run "$RUN" --selection "$SEL" --output "$OUT" --methods Geometry TrackedPGVB StableTrackedPGVB
$PY -m scripts.audit_stable_navigation_rollouts --run "$RUN" --previous-run /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/landmark_acquisition_20261105/full64 --selection "$SEL" --output "$OUT"
$PY -m scripts.audit_landmark_acquisition_artifacts --run "$RUN" --selection "$SEL" --output "$OUT"
$PY -m scripts.plot_acquisition_transitions --run "$RUN" --output "$OUT" --highlight StableTrackedPGVB --indices 0 1 10 13 18 21 24 27 33 34 39 43 49
