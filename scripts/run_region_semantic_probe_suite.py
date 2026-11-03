"""Persistent smoke/recovery then fixed-final region probe, no navigation."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

from graphnav.checkpointing import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Preserve prior suite')
    args.output.mkdir(parents=True)
    common = [sys.executable, '-m', 'scripts.train_region_semantic_probe',
        '--cache', str(args.run_root/'chain_train400_features'),
        '--labels', str(args.run_root/'region_semantic_probe_labels_v1'),
        '--native-text', str(args.run_root/'region_semantic_text_v1.pt')]
    stages = [
        ('smoke', ['--output', str(args.output/'smoke'), '--epochs', '1', '--max-batches', '2']),
        ('resume-smoke', ['--output', str(args.output/'smoke'), '--epochs', '2', '--max-batches', '2', '--resume']),
        ('pilot', ['--output', str(args.output/'pilot')]),
    ]
    status = dict(status='running', stages=[], started_utc=datetime.now(timezone.utc).isoformat())
    for name, command in stages:
        status['current_stage'] = name
        atomic_json(status, args.output/'status.json')
        with (args.output/(name+'.log')).open('x') as stream:
            result = subprocess.run(common+command, stdout=stream, stderr=subprocess.STDOUT)
        status['stages'].append(dict(stage=name, returncode=result.returncode, ended_utc=datetime.now(timezone.utc).isoformat()))
        if result.returncode:
            status['status'] = 'failed'; atomic_json(status, args.output/'status.json')
            raise SystemExit(result.returncode)
    status['status'] = 'complete'; atomic_json(status, args.output/'status.json')


if __name__ == '__main__': main()
