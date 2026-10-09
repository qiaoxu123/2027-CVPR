"""Summarize completed paired rollouts; no inference, selection, or GT input."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


KEYS = ('SR', 'SPL', 'OSR', 'NE', 'nDTW', 'sDTW', 'steps', 'inference_ms')


def read_run(path):
    path = Path(path)
    report = json.loads((path / 'metrics.json').read_text())
    rows = [json.loads(s) for s in (path / 'predictions.jsonl').read_text().splitlines()]
    indices = [r['index'] for r in rows]
    if len(set(indices)) != len(indices) or len(rows) != report['metrics']['episodes']:
        raise ValueError('duplicate or incomplete rollout denominator')
    digest = hashlib.sha256((path / 'episodes.json').read_bytes()).hexdigest()
    if digest != report['episode_manifest_sha256']:
        raise ValueError('episode manifest changed')
    return report, {r['index']: r for r in rows}, digest


def paired_stats(base, candidate, seed=20261010, draws=2000):
    if base.keys() != candidate.keys():
        raise ValueError('paired episode IDs differ')
    keys = sorted(base)
    for i in keys:
        if base[i]['episode_id'] != candidate[i]['episode_id'] or base[i]['poses'][0] != candidate[i]['poses'][0]:
            raise ValueError('paired source episode or start differs')
    rng = np.random.default_rng(seed)
    differences = np.array([[candidate[i]['metrics'][k] - base[i]['metrics'][k] for k in KEYS] for i in keys])
    means = np.empty((draws, len(KEYS)))
    for j in range(draws):
        means[j] = differences[rng.integers(0, len(keys), len(keys))].mean(0)
    maps = sorted({base[i]['map_name'] for i in keys})
    map_differences = np.array([differences[[j for j, i in enumerate(keys) if base[i]['map_name'] == name]].mean(0) for name in maps])
    map_means = np.empty_like(means)
    for j in range(draws):
        map_means[j] = map_differences[rng.integers(0, len(maps), len(maps))].mean(0)
    gained = [i for i in keys if candidate[i]['metrics']['SR'] > base[i]['metrics']['SR']]
    lost = [i for i in keys if candidate[i]['metrics']['SR'] < base[i]['metrics']['SR']]
    return dict(episodes=len(keys), gained=gained, lost=lost,
        episode_weighted={k: dict(delta=float(differences[:, j].mean()), ci95=np.quantile(means[:, j], [.025, .975]).tolist()) for j, k in enumerate(KEYS)},
        equal_map_weighted={k: dict(delta=float(map_differences[:, j].mean()), ci95=np.quantile(map_means[:, j], [.025, .975]).tolist()) for j, k in enumerate(KEYS)},
        uncertainty='Paired episode bootstrap and paired equal-map bootstrap; these estimate different weighting schemes. Single training seed, not training-seed uncertainty.',
        bootstrap_seed=seed, draws=draws)


def failures(rows):
    values = list(rows.values())
    return dict(episodes=len(values), failed=sum(not r['metrics']['SR'] for r in values),
        visited_success_but_final_failed=sum(r['metrics']['OSR'] and not r['metrics']['SR'] for r in values),
        failed_policy_stops=sum(r['termination'] == 'policy_stop' and not r['metrics']['SR'] for r in values),
        failed_horizon=sum(r['termination'] == 'horizon' and not r['metrics']['SR'] for r in values),
        failed_stagnation=sum(r['termination'] == 'stagnation' and not r['metrics']['SR'] for r in values),
        nonadjacent_revisit_episodes=sum(any(np.linalg.norm(np.asarray(r['poses'])[j, :2] - np.asarray(r['poses'])[i, :2]) < 2 for j in range(2, len(r['poses'])) for i in range(j - 1)) for r in values),
        note='Failure counts can overlap. XY revisits use <2m separation and skip adjacent poses; altitude descent at a fixed XY can count as a revisit, so this is not automatically a control oscillation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True, help='name=completed_run_directory; first is paired baseline')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError('refusing analysis overwrite')
    runs = {}
    for spec in args.run:
        name, path = spec.split('=', 1)
        if name in runs:
            raise ValueError('duplicate model name')
        runs[name] = (path, *read_run(path))
    baseline = next(iter(runs))
    if len({r[3] for r in runs.values()}) != 1:
        raise ValueError('paired manifests differ')
    output.mkdir(parents=True)
    summary = dict(baseline=baseline, episode_manifest_sha256=runs[baseline][3], models={}, paired={})
    table, maps, episodes = [], [], []
    for name, (path, report, rows, _) in runs.items():
        summary['models'][name] = dict(path=path, metrics=report['metrics'], failures=failures(rows),
            seconds=report['seconds'], peak_allocated_gib=report['gpu_peak_allocated_gib'],
            checkpoint_sha256=report['checkpoint_sha256'], head_checkpoint_sha256=report.get('head_checkpoint_sha256'),
            dynamic_checkpoint_sha256=report.get('dynamic_checkpoint_sha256'), graphnav_commit=report['graphnav_commit'])
        table.append(dict(model=name, **{k: report['metrics'].get(k) for k in (*KEYS, 'False_Stop_Rate')}, episodes=len(rows), seconds=report['seconds']))
        maps.extend(dict(model=name, map_name=m, **{k: v[k] for k in KEYS if k in v}, episodes=v['episodes']) for m, v in report['by_map'].items())
        episodes.extend(dict(model=name, index=i, map_name=r['map_name'], termination=r['termination'], **{k: r['metrics'][k] for k in KEYS}) for i, r in rows.items())
        if name != baseline:
            summary['paired'][name] = paired_stats(runs[baseline][2], rows)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    for name, records in (('comparison', table), ('maps', maps), ('episodes', episodes)):
        with (output / (name + '.csv')).open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    print(json.dumps(dict(output=str(output), episodes=len(runs[baseline][2]), models=list(runs))))


if __name__ == '__main__':
    main()
