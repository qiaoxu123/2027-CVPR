"""Export measured training histories/configuration without inferring missing data."""
import argparse
import csv
import json
from pathlib import Path

from graphnav.checkpointing import file_sha256, provenance


def summarize(specifications, output):
    output = Path(output)
    if output.exists():
        raise ValueError('refusing training report overwrite')
    rows, stages = [], {}
    for specification in specifications:
        name, folder = specification.split('=', 1)
        if name in stages:
            raise ValueError('duplicate training stage: ' + name)
        folder = Path(folder)
        report_path = folder / 'report.json'
        report = json.loads(report_path.read_text())
        history = report['history']
        stage_rows = []
        for epoch in history:
            for split, metrics in epoch.items():
                if not isinstance(metrics, dict) or 'seconds' not in metrics:
                    continue
                row = dict(stage=name, run=str(folder), epoch=epoch['epoch'], split=split,
                           started_unix=metrics.get('started_unix'),
                           finished_unix=metrics.get('finished_unix'))
                row.update({key: value for key, value in metrics.items()
                            if value is None or isinstance(value, (int, float, bool, str))})
                stage_rows.append(row)
        if not stage_rows:
            raise ValueError('no measured epoch durations: ' + name)
        rows.extend(stage_rows)
        stages[name] = dict(run=str(folder), report_sha256=file_sha256(report_path),
                            args=report.get('args'), provenance=report.get('provenance'),
                            input_fingerprint=report.get('input_fingerprint'),
                            epochs=len(history),
                            train_seconds=sum(row['seconds'] for row in stage_rows if row['split'] == 'train'),
                            evaluation_seconds=sum(row['seconds'] for row in stage_rows if row['split'] != 'train'),
                            epochs_with_wall_clock=sum(row['started_unix'] is not None for row in stage_rows),
                            checkpoints={p.name: dict(path=str(p), sha256=file_sha256(p))
                                         for p in sorted(folder.glob('*.pt')) if p.name in
                                         ('best.pt', 'last.pt', 'pretrain_best.pt', 'pretrain_last.pt', 'visual_best.pt', 'visual_last.pt')})
    output.mkdir(parents=True)
    with (output / 'epochs.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(stages=stages, provenance=provenance(), note='Measured epoch durations only; training totals exclude setup, rendering and feature-cache generation, which are recorded as separate queue jobs. Epochs without absolute timestamps cannot be matched to sampled GPU telemetry. Missing values remain empty/null. Nested per-query/class metrics remain in source reports; no synthetic reconstruction.')
    (output / 'summary.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True, help='stage=completed training directory')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = summarize(args.run, args.output)
    print(json.dumps(dict(output=args.output, stages=len(result['stages']))))


if __name__ == '__main__':
    main()
