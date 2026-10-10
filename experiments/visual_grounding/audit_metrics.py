"""Postprocess frozen predictions: object balance, category counts, chance, timing.

This analysis never encodes images, changes labels, or selects configurations.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.visual_grounding.freeze_dataset import read
from experiments.visual_grounding.statistics import intervals, macro_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='runs/visual_grounding_benchmark')
    args = parser.parse_args()
    root = Path(args.root)
    results = {}
    names = ['remoteclip_vitl14', 'remoteclip_vitb32', 'regionclip_rn50', 'siglip2_large384']
    for name in names:
        groups = defaultdict(list)
        for file in (root / name).glob('*/predictions_*.jsonl'):
            if file.name not in ['predictions_development.jsonl', 'predictions_unseen.jsonl']:
                continue
            for row in read(file):
                groups[row['split'], row['task'], row['category']].append(row)
        summary = {}
        for key, rows in sorted(groups.items()):
            objects = defaultdict(list)
            for row in rows:
                objects[row['map_name'], row['object_id']].append(float(row['correct']))
            summary['/'.join(key)] = {
                'events': len(rows), 'independent_instances': len(objects),
                'object_balanced': intervals([np.mean(v) for v in objects.values()],
                                             [k[0] for k in objects]),
            }
            if all('truth' in row and 'prediction' in row for row in rows):
                summary['/'.join(key)]['attribute_macro_f1'] = macro_f1(
                    [row['truth'] for row in rows], [row['prediction'] for row in rows])
        results[name] = summary
    chance = {}
    for folder in (root / 'data_audit').glob('frozen_*'):
        if not (folder / 'frozen.json').exists():
            continue
        queries = read(folder / 'queries.jsonl')
        categories = defaultdict(list)
        for q in queries:
            categories[q['attribute_type']].append(q)
        chance[folder.name] = {
            'queries': len(queries), 'multiple_positive_queries': sum(len(q['positive_sample_ids']) > 1 for q in queries),
            'random_r1': float(np.mean([len(q['positive_sample_ids']) / len(q['candidate_sample_ids']) for q in queries])) if queries else None,
            'by_attribute': {kind: {'queries': len(rows),
                'candidate_objects': len({(q['map_name'], obj) for q in rows for obj in q['candidate_object_ids']}),
                'random_r1': float(np.mean([len(q['positive_sample_ids']) / len(q['candidate_sample_ids']) for q in rows]))}
                for kind, rows in categories.items()},
        }
    output = root / 'comparisons'
    output.mkdir(exist_ok=True)
    (output / 'object_balanced_metrics.json').write_text(json.dumps(results, indent=2))
    (output / 'candidate_chance.json').write_text(json.dumps(chance, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    runtime = [json.loads((root / n / 'runtime_benchmark.json').read_text()) for n in names]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, batch, field, label in zip(axes, ['1', '16', '16'], ['median_ms', 'median_ms', 'peak_allocated_gib'],
                                     ['Single ROI + text latency (ms)', '16 candidates + text latency (ms)', 'Peak allocated GPU memory (GiB)']):
        ax.bar(range(4), [r['results'][batch][field] for r in runtime], color=['#187bcd', '#62a6d6', '#d98c27', '#54a16c'])
        ax.set_xticks(range(4), ['Remote L', 'Remote B', 'Region RN50', 'SigLIP2'], rotation=30)
        ax.set_ylabel(label)
    fig.suptitle('Serial CUDA measurement; decoded real pixels; feature caches cleared')
    fig.tight_layout()
    fig.savefig(output / 'isolated_latency_memory.png', dpi=180)
    plt.close(fig)
    print(json.dumps({'models': len(results), 'splits': list(chance)}))


if __name__ == '__main__':
    main()
