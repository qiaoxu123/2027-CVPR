#!/usr/bin/env bash
# Serial execution adapter for the already checked, pinned RTX5090 V4 tree.
set -euo pipefail

TASK_ROOT=/mnt/hdd8t/Workspace/2027-CVPR-A2-20261028
LEGACY_ROOT="$TASK_ROOT/legacy_home/Workspace/2027-CVPR/Experiments/A2WaldoLandmarkBelief"
OUTPUT="$LEGACY_ROOT/runs/a2_waldo_landmark_20261027/revision_v4"
# Keep frozen cache path identities while exposing only the new code inside
# this process. Neither the old source tree nor its run files are rewritten.
GUEST_ROOT=/home/20261024_1/Workspace/2027-CVPR/Experiments/A2WaldoLandmarkBelief
SOURCE_TREE="$TASK_ROOT/legacy_home/Workspace/2027-CVPR/Experiments/A2WaldoLandmarkBeliefV4"
PYTHON=/home/20261024_1/miniconda3/envs/graphnav/bin/python
PROOT="$TASK_ROOT/tools/root/usr/bin/proot"
LOGS="$OUTPUT/pipeline_logs"
PHASE=startup
mkdir -p "$LOGS"

status() {
  /usr/bin/python3 - "$OUTPUT" "$1" "$PHASE" "$2" "$$" "$0" <<'PY'
import datetime,hashlib,json,pathlib,sys
directory=pathlib.Path(sys.argv[1]);script=pathlib.Path(sys.argv[6])
value=dict(state=sys.argv[2],phase=sys.argv[3],exit_code=int(sys.argv[4]),launcher_pid=int(sys.argv[5]),
           timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
           scientific_commit='ac03d24dde96c7ebe0bfe22e2a1e8196c0d1ea96',
           execution_script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
           independent_calibration_pending=True,full_navigation_pending=True,objective_achieved=False)
marker=directory/'training/training_complete.json'
if marker.exists():
    completed=json.loads(marker.read_text())
    value.update(selected_epoch=completed['selected_epoch'],checkpoint_eligible=completed['checkpoint_eligible'])
temporary=directory/'pipeline_status.tmp';temporary.write_text(json.dumps(value,indent=2)+'\n')
temporary.replace(directory/'pipeline_status.json')
PY
}

finish() {
  code=$?
  trap - EXIT
  if (( code != 0 )); then status failed "$code"; fi
  exit "$code"
}
trap finish EXIT

guest() {
  env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    HF_HOME=/home/20261024_1/.cache/huggingface \
    HF_HUB_CACHE=/home/20261024_1/.cache/huggingface/hub \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    LD_LIBRARY_PATH="$TASK_ROOT/tools/root/usr/lib/x86_64-linux-gnu" \
    "$PROOT" -b "$TASK_ROOT/legacy_home:/home/20261024_1" \
    -b "$SOURCE_TREE:$GUEST_ROOT" -b "$LEGACY_ROOT/runs:$GUEST_ROOT/runs" -w "$GUEST_ROOT" \
    "$PYTHON" -u "$@"
}

stage() {
  PHASE="$1"; shift
  status running 0
  guest "$@" >> "$LOGS/$PHASE.log" 2>&1
}

# The actual remote CPU gate is already complete. Do not rerun it and change
# its log/proof hashes during a resumed training run.
stage cpu_gate_check -c 'from experiments.a2_waldo_landmark_belief.reference_v4_preparation import current_binding; current_binding(); print("Current V4 CPU/data/assets proof verified",flush=True)'

if [[ ! -f "$OUTPUT/training/training_complete.json" ]]; then
  if [[ ! -f "$OUTPUT/training/last.pt" ]]; then
    stage cuda -m experiments.a2_waldo_landmark_belief.reference_v4_preparation --cuda
    stage runtime_a0 -m experiments.a2_waldo_landmark_belief.reference_v4_preflight --arm a0
    stage runtime_off -m experiments.a2_waldo_landmark_belief.reference_v4_preflight --arm off
    stage runtime_on -m experiments.a2_waldo_landmark_belief.reference_v4_preflight --arm on
    stage runtime_verify -m experiments.a2_waldo_landmark_belief.reference_v4_preflight --arm verify
  fi
  stage training -m experiments.a2_waldo_landmark_belief.reference_v4_train --resume
fi

if [[ -f "$OUTPUT/training/artifact_verification.json" ]]; then
  stage verification_check -c 'from experiments.a2_waldo_landmark_belief.reference_v4_train import DIRECTORY,training_signature; from experiments.a2_waldo_landmark_belief.reference_v4_verify import validate_verified_proof; validate_verified_proof(DIRECTORY,training_signature()); print("Existing complete V4 training verification retained",flush=True)'
else
  stage verification -m experiments.a2_waldo_landmark_belief.reference_v4_verify
fi
stage curves -m experiments.a2_waldo_landmark_belief.reference_v4_verify --curves
PHASE=full_training_verified
status complete 0
