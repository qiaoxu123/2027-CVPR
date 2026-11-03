"""Persistent sequential GPU smoke/resume and equal-budget TRAIN pilots."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from graphnav.checkpointing import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Do not rerun or overwrite a suite')
    args.output.mkdir(parents=True)
    root = args.run_root
    common = [sys.executable, '-m', 'scripts.train_world_relation_factor',
        '--cache', str(root/'chain_train400_features'),
        '--fit-labels', str(root/'fixed_yaw_train400_s1/labels_fit.jsonl'),
        '--dev-labels', str(root/'fixed_yaw_train400_s1/labels_dev.jsonl'),
        '--fit-observation-filter', str(root/'public_deployment_filter.json'),
        '--base-checkpoint', str(root/'world_v3_envelope_s0/best.pt'),
        '--base-role-vectors', str(root/'reference_vectors_v2.pt'),
        '--roles', str(root/'relation_factor_roles_v1/roles.json'),
        '--reference-vectors', str(root/'relation_factor_vectors_v1.pt'),
        '--catalog', '/home/20261024_1/Workspace/DATA/sbfnav-data/cityrefer/objects.json']
    smoke = args.output/'smoke'
    stages = [
        ('smoke', ['--relation-mode', 'typed', '--output', str(smoke), '--epochs', '1', '--steps-per-epoch', '2']),
        ('resume-smoke', ['--relation-mode', 'typed', '--output', str(smoke), '--epochs', '2', '--steps-per-epoch', '2', '--resume']),
        ('hidden', ['--relation-mode', 'hidden', '--output', str(args.output/'hidden_s0')]),
        ('typed', ['--relation-mode', 'typed', '--output', str(args.output/'typed_s0')]),
    ]
    progress = dict(status='running', stages=[], started_utc=datetime.now(timezone.utc).isoformat())
    for name, arguments in stages:
        command = common+arguments
        progress.update(current_stage=name, current_command=command)
        atomic_json(progress, args.output/'status.json')
        with (args.output/(name+'.log')).open('x') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        progress['stages'].append(dict(stage=name, returncode=result.returncode,
                                      ended_utc=datetime.now(timezone.utc).isoformat()))
        if result.returncode:
            progress['status'] = 'failed'
            atomic_json(progress, args.output/'status.json')
            raise SystemExit(result.returncode)
    progress['status'] = 'complete'
    atomic_json(progress, args.output/'status.json')
    print(json.dumps(progress, indent=2), flush=True)


if __name__ == '__main__':
    main()
