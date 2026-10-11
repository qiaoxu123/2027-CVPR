"""Additional Seen-only Stage1 controls and exact initial-controller replay audit."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from graphnav.checkpointing import atomic_json, file_sha256
from scripts.summarize_closedloop import read_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='runs/graphnav_v1_stage1')
    args = parser.parse_args()
    root = Path(args.output).resolve()
    base = json.loads((root/'baseline_seen128/metrics.json').read_text())['args']
    status = {'status': 'running', 'jobs': {}, 'pid': os.getpid()}
    env = dict(os.environ, HF_HUB_OFFLINE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    jobs = [('initial_seen128', root/'initial_complete.pt', 'real', 'current')]
    jobs += [('stage_A_'+control+'_seen128', root/'stages/A/best.pt', control, 'current') for control in ('real', 'zero', 'shuffle', 'no-belief')]
    jobs += [('stage_B_'+control+'_seen128', root/'stages/B/best.pt', 'real', control) for control in ('current', 'history')]
    try:
        for name, checkpoint, visual, history in jobs:
            stage = checkpoint.parent
            if stage.name in ('A', 'B'):
                status.update(stage=name, status='waiting_stage_completion')
                atomic_json(status, root/'audit_status.json')
                while not (stage/'complete.json').exists():
                    time.sleep(5)
            status.update(stage=name, status='running')
            atomic_json(status, root/'audit_status.json')
            destination = root/name
            if not (destination/'metrics.json').exists():
                if destination.exists():
                    destination.rename(root/(name+'_incomplete_'+str(int(time.time()))))
                command = [sys.executable, '-m', 'scripts.evaluate_closedloop', '--model', 'graphnav-v1',
                    '--sbf-source', base['sbf_source'], '--config', base['config'], '--data-root', base['data_root'],
                    '--checkpoint', str(checkpoint), '--split', 'val_seen', '--max-episodes', '128',
                    '--output', str(destination), '--stage1-visual-control', visual, '--stage1-history-control', history,
                    '--evaluation-concurrency', '2']
                with (root/(name+'.log')).open('a') as stream:
                    subprocess.run(command, check=True, env=env, stdout=stream, stderr=subprocess.STDOUT)
            report = json.loads((destination/'metrics.json').read_text())
            if report['checkpoint_sha256'] != file_sha256(checkpoint):
                raise ValueError('control checkpoint changed: '+name)
            status['jobs'][name] = dict(status='passed', checkpoint_sha256=report['checkpoint_sha256'])
            atomic_json(status, root/'audit_status.json')
        baseline, rows, digest = read_run(root/'baseline_seen128')
        candidate, replay, other_digest = read_run(root/'initial_seen128')
        if digest != other_digest or rows.keys() != replay.keys():
            raise ValueError('initial complete replay episode manifest differs')
        errors = []
        maximum = 0.
        for index, row in rows.items():
            other = replay[index]
            exact = row['poses'] == other['poses'] and row['termination'] == other['termination']
            same_decisions = len(row['step_diagnostics']) == len(other['step_diagnostics'])
            for a, b in zip(row['step_diagnostics'], other['step_diagnostics']):
                same_decisions &= all(a.get(k) == b.get(k) for k in ('candidate','mode','stop_probability','selected_goal_xy'))
            if len(row['poses']) == len(other['poses']):
                maximum = max(maximum, float(np.abs(np.asarray(row['poses'])-np.asarray(other['poses'])).max()))
            if not exact or not same_decisions:
                errors.append(index)
        audit = dict(episodes=len(rows), episode_manifest_sha256=digest, poses_stop_candidates_exact=not errors,
            differing_episode_indices=errors, max_pose_difference=maximum, sources=json.loads((root/'initial_sources.json').read_text()),
            baseline_protocol=baseline['protocol'], complete_protocol=candidate['protocol'], baseline_metrics=baseline['metrics'], complete_metrics=candidate['metrics'],
            timing_note='Supplementary Seen controls overlap training; timing is measured under concurrency=2 and cannot establish isolated speedup.')
        atomic_json(audit, root/'initial_seen128_parity.json')
        if errors or baseline['protocol'] != candidate['protocol']:
            raise ValueError('initial complete system failed exact strong-baseline replay')
        status.update(status='complete', finished=time.time())
    except BaseException as exc:
        status.update(status='failed', error=repr(exc))
        raise
    finally:
        atomic_json(status, root/'audit_status.json')


if __name__ == '__main__':
    main()
