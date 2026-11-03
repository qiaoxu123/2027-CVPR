#!/bin/bash
set -e
trap 'printf "%s\n" "$?" > /tmp/public_geometry_visual/exit_code' EXIT
cd /home/20261024_1/Workspace/2027-CVPR/Experiments/GeomBeliefLandmarkChain
R=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/landmark_chain_visual
PY=/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python
$PY -m scripts.prepare_public_goal_geometry --cache $R/chain_train400_features --observations $R/fixed_yaw_train400_s1 --cache $R/scoped_between_train294_features_v1 --observations $R/scoped_between_train294_observations_v1 --catalog /home/20261024_1/Workspace/DATA/sbfnav-data/cityrefer/objects.json --corpus $R/scoped_between_train_v2/records.json --output $R/public_geometry_visual_fields_v1
$PY -m scripts.train_public_geometry_visual --root $R --geometry $R/public_geometry_visual_fields_v1/geometry_public.pt --output $R/public_geometry_visual_suite_v1_s0/smoke --mode conditioned --steps 2
$PY -m scripts.train_public_geometry_visual --root $R --geometry $R/public_geometry_visual_fields_v1/geometry_public.pt --output $R/public_geometry_visual_suite_v1_s0/smoke --mode conditioned --steps 4 --resume
for mode in map unconditioned conditioned; do
$PY -m scripts.train_public_geometry_visual --root $R --geometry $R/public_geometry_visual_fields_v1/geometry_public.pt --output $R/public_geometry_visual_suite_v1_s0/$mode --mode $mode
done
$PY - <<'PY'
import json
from pathlib import Path
from graphnav.checkpointing import file_sha256
p=Path('/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/landmark_chain_visual/public_geometry_visual_suite_v1_s0')
(p/'freeze.json').write_text(json.dumps(dict(checkpoints={str(p/m/'final.pt'):file_sha256(p/m/'final.pt') for m in ('map','unconditioned','conditioned')},dev_labels_read_by_suite_before_freeze=False),indent=2))
PY
for mode in map unconditioned conditioned; do
$PY -m scripts.train_public_geometry_visual --root $R --geometry $R/public_geometry_visual_fields_v1/geometry_public.pt --output $R/public_geometry_visual_suite_v1_s0/$mode --mode $mode --evaluate --freeze-manifest $R/public_geometry_visual_suite_v1_s0/freeze.json
done
