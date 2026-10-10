"""Finish interrupted controls and run paired E2 without overwriting prior runs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.run_static_controlled_queue import BASE, ROOT


def main():
    output = ROOT / 'runs/static_completed_20261010'
    output.mkdir(exist_ok=False)
    env = dict(os.environ, PYTHONPATH=str(ROOT), HF_HUB_OFFLINE='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    recipes = [('A3_relation000', ['--relation-auxiliary', '--relation-weight', '0']),
               ('A3_relation001', ['--relation-auxiliary', '--relation-weight', '0.01']),
               ('A3_relation005', ['--relation-auxiliary', '--relation-weight', '0.05']),
               ('A4_reliable_relation', ['--variant', 'gat', '--graph-control', 'reliable-relation'])]
    status = {'jobs': {}, 'selection': 'Seen R@1 then lower NE; Unseen untouched'}

    def run(name, command):
        status['jobs'][name] = dict(command=command, started=time.time(), status='running')
        (output / 'status.json').write_text(json.dumps(status, indent=2))
        with (output / (name + '.log')).open('w') as stream:
            code = subprocess.call(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        status['jobs'][name].update(exit_code=code, finished=time.time(), status='passed' if code == 0 else 'failed')
        (output / 'status.json').write_text(json.dumps(status, indent=2))
        if code:
            raise RuntimeError(name + ' failed')

    for name, flags in recipes:
        run(name + '_smoke', BASE + flags + ['--epochs', '1', '--max-batches', '1',
                                           '--output', str(output / (name + '_smoke'))])
    for name, flags in recipes:
        run(name, BASE + flags + ['--epochs', '5', '--output', str(output / name)])
    path = ROOT / 'runs/static_E2_E3_20261010/A4_relation'
    run('A4_relation_resume', BASE + ['--variant', 'gat', '--graph-control', 'relation',
                                    '--epochs', '5', '--output', str(path), '--resume', str(path / 'last.pt')])
    status['completed'] = time.time()
    (output / 'status.json').write_text(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
