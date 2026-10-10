"""Finish the ordered A4 batch sweep after an already-running batch-64 job."""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def active(pid):
    path = Path(f"/proc/{pid}/stat")
    return path.exists() and path.read_text().split(") ", 1)[1].split()[0] != "Z"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wait-pid", type=int, required=True)
    args = p.parse_args()
    while active(args.wait_pid):
        time.sleep(2)
    assert Path("runs/a4_batch64_workers4/full_localization_metrics.csv").exists()
    environment = dict(os.environ, PYTHONPATH=".", HF_HUB_OFFLINE="1",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    history = []
    for batch_size in (128, 256):
        command = [sys.executable, "-m", "graphnav.full_experiment",
            "--train", "data/revised_static/graphnav_train_seen.jsonl",
            "--val-seen", "data/revised_static/graphnav_val_seen.jsonl",
            "--val-unseen", "data/revised_static/graphnav_val_unseen.jsonl",
            "--dataset-version", "revised", "--match-mode", "fuzzy",
            "--variants", "gat", "--epochs", "5", "--batch-size", str(batch_size),
            "--workers", "4", "--output", f"runs/a4_batch{batch_size}_workers4"]
        start = datetime.datetime.now(datetime.timezone.utc).isoformat()
        print(json.dumps({"started_batch": batch_size, "time": start}), flush=True)
        with Path(f"runs/local_validation/a4_batch{batch_size}.log").open("w") as log:
            completed = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        history.append({"batch_size": batch_size, "started_utc": start,
            "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "returncode": completed.returncode, "command": command})
        Path("runs/local_validation/a4_followups.json").write_text(json.dumps(history, indent=2))
        if completed.returncode:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
