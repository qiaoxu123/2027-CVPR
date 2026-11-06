#!/bin/bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR-relation-bearing
RUN=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/relation_bearing_s42_20261106
export OMP_NUM_THREADS=4
"$RUN/parser_env/bin/python" -u -m scripts.prepare_relation_language --data-manifest /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/polygon_neighborhood_bearing_s42_20261106/full/data_manifest.json --public-objects /home/20261024_1/Workspace/DATA/sbfnav-data/cityrefer_original/objects.json --old-protocol /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/polygon_neighborhood_bearing_s42_20261106/full/protocol.json --output "$RUN/data" > "$RUN/parse.log" 2>&1
