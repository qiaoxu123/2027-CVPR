"""Wait for all resumed splits, then compare with the frozen formal 28 grid."""
import csv
import hashlib
import io
import json
from pathlib import Path
import time

from graphnav.checkpointing import atomic_json
from graphnav.workspace_paths import PROJECT


ROOT = Path(__file__).resolve().parents[1]
SPLITS = {'val_seen': ('full_seen', 2470),
          'val_unseen': ('full_unseen', 2697),
          'test_unseen': ('full_test_unseen', 5281)}


def summarize():
    rows = []
    frozen = json.loads((ROOT / 'full_resolution_384/frozen_resolution.json').read_text())
    formal = PROJECT / 'GraphNav/runs/graphnav_v1_stage1'
    for split, (baseline, expected) in SPLITS.items():
        directory = ROOT / 'full_resolution_384' / split
        high = json.loads((directory / 'metrics.json').read_text())
        low = json.loads((formal / baseline / 'metrics.json').read_text())
        assert high['metrics']['episodes'] == low['metrics']['episodes'] == expected
        assert high['checkpoint_sha256'] == low['checkpoint_sha256'] == frozen['checkpoint_sha256']
        assert high['args']['stage1_grid_size'] == 384
        assert high['args']['stop_threshold'] == low['args']['stop_threshold'] == .3
        manifest = directory / 'episodes.json'
        assert json.loads(manifest.read_text()) == json.loads((formal / baseline / 'episodes.json').read_text())
        assert high['episode_manifest_sha256'] == low['episode_manifest_sha256']
        assert hashlib.sha256(manifest.read_bytes()).hexdigest() == high['episode_manifest_sha256']
        with (directory / 'predictions.jsonl').open() as stream:
            # Preserve large traces; only inspect each row's episode index.
            for count, line in enumerate(stream, 1):
                assert json.loads(line)['index'] == count - 1
        assert count == expected
        row = {'split': split, 'episodes': expected}
        for metric, scale, unit in [('SR', 100, 'pp'), ('SPL', 100, 'pp'), ('NE', 1, 'm')]:
            a = low['metrics'][metric] * scale
            b = high['metrics'][metric] * scale
            row.update({f'{metric}_28': a, f'{metric}_384': b, f'{metric}_delta_{unit}': b-a})
        row.update(checkpoint_sha256=high['checkpoint_sha256'],
                   episode_manifest_sha256=high['episode_manifest_sha256'])
        rows.append(row)
    target = ROOT / 'full_resolution_384'
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    temporary = target / 'COMPARISON_28_VS_384.csv.tmp'
    temporary.write_text(buffer.getvalue())
    temporary.replace(target / 'COMPARISON_28_VS_384.csv')
    atomic_json({'complete': True, 'finished': time.time(), 'rows': rows,
                 'units': 'SR/SPL percent; delta percentage points; NE/delta meters',
                 'timing_note': 'Evaluator seconds describe the final invocation, not cumulative runtime across pauses.'},
                target / 'comparison_complete.json')
    print(buffer.getvalue(), flush=True)


def main():
    print('Waiting for all three complete metrics; no partial-prefix comparison.', flush=True)
    while not all((ROOT / 'full_resolution_384' / split / 'metrics.json').exists() for split in SPLITS):
        queue = ROOT / 'full_resolution_384/parallel_queue.json'
        if queue.exists() and any(job['status'] == 'failed' for job in json.loads(queue.read_text())):
            raise RuntimeError('Evaluation failed; preserve outputs and inspect split logs.')
        time.sleep(30)
    summarize()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        atomic_json({'error': str(error), 'time': time.time()},
                    ROOT / 'full_resolution_384/comparison_failure.json')
        raise
