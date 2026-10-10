"""Execute the fixed core evaluation as each completed epoch-3 head becomes ready.

This runner never trains, changes configuration or opens Unseen. All commands and
exit codes are saved. Reports use only completed actual evaluation artifacts.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def run(args):
    root = Path(args.root)
    cache = Path(args.cache_root) if args.cache_root else root/'cache_fp16'
    records = []
    logs = root/'evaluation_logs'; logs.mkdir(parents=True, exist_ok=True)

    def execute(stage, arguments):
        command = [sys.executable, '-m', *arguments]
        started = time.time()
        print(json.dumps(dict(stage=stage, command=command, started=started)), flush=True)
        with (logs/f'{stage}.log').open('w') as handle:
            process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
        records.append(dict(stage=stage, command=command, exit_code=process.returncode,
            started_utc_seconds=started, elapsed_seconds=time.time()-started))
        (root/'evaluation_commands.json').write_text(json.dumps(records, indent=2))
        if process.returncode:
            raise RuntimeError(f'{stage} failed; see {logs/stage}.log')

    waiting_since = time.time()
    for condition in ('E0','E1','E2'):
        complete = root/condition/'training_complete.json'
        while not complete.exists():
            if time.time()-waiting_since > args.timeout_seconds:
                raise TimeoutError('completed training checkpoint not available')
            time.sleep(10)
        payload = json.loads(complete.read_text())
        if payload['completed_epochs'] != 3 or payload['global_steps'] != 1920:
            raise ValueError('completed training does not match frozen three-epoch budget')
        execute('navigation_'+condition, ['experiments.bidirectional_core.runtime',
            '--episodes', str(root/'data/val_seen/pilot_episodes.jsonl'),
            '--checkpoint', str(root/condition/'epoch03.pt'),
            '--static-cache', str(cache/'val_seen'),
            '--output', str(root/'navigation'/condition)])
    arguments = ['experiments.bidirectional_core.evaluate', '--cache', str(cache/'val_seen'),
        '--output', str(root/'offline')]
    for condition in ('E0','E1','E2'):
        arguments += ['--checkpoint', condition+'='+str(root/condition/'epoch03.pt')]
    execute('offline', arguments)
    execute('report', ['experiments.bidirectional_core.report', '--offline', str(root/'offline/metrics.json'),
        '--navigation', str(root/'navigation'), '--output', str(root/'comparisons'),
        '--training-root', str(root), '--data-root', str(root/'data'), '--doc', args.doc])
    execute('cases', ['experiments.bidirectional_core.cases', '--cache', str(cache/'val_seen'),
        '--predictions', str(root/'offline'), '--output', str(root/'cases')])
    (root/'evaluation_complete.json').write_text(json.dumps(dict(complete=True,
        completed_seconds=time.time(), stages=records, unseen_opened=False), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='runs/bidirectional_core_test')
    parser.add_argument('--doc', default='docs/BIDIRECTIONAL_CORE_TEST_REPORT.md')
    parser.add_argument('--cache-root', help='numerically verified frozen cache, default ROOT/cache_fp16')
    parser.add_argument('--timeout-seconds', type=int, default=14400)
    run(parser.parse_args())


if __name__ == '__main__': main()
