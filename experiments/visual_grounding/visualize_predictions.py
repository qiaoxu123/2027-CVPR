"""Render reproducible Seen-only model agreement/failure cases from cached scores."""
import argparse
import json
import textwrap
from pathlib import Path

from PIL import Image

from experiments.visual_grounding.freeze_dataset import read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='runs/visual_grounding_benchmark')
    args = parser.parse_args()
    root = Path(args.root)
    data = root / 'data_audit/frozen_seen'
    index = {r['sample_id']: r for r in read(data / 'index.jsonl')}
    queries = read(data / 'queries.jsonl')
    names = ['remoteclip_vitl14', 'remoteclip_vitb32', 'regionclip_rn50', 'siglip2_large384']
    predictions = [{(r['query_id'], r['category']): r for r in read(root / n / 'seen/predictions_development.jsonl')
                    if r['task'] == 'weak_hard_negative_attribute_phrase_r1'} for n in names]
    selected = []
    seen = set()
    for criterion in ['all_fail', 'siglip_only', 'region_only', 'all_pass', 'roof_color', 'vehicle_color']:
        added = 0
        for q in queries:
            key = q['query_sample_id'], q['attribute_type']
            if key in seen:
                continue
            successes = [bool(p[key]['correct']) for p in predictions]
            valid = {'all_fail': not any(successes), 'siglip_only': successes == [False, False, False, True],
                     'region_only': successes == [False, False, True, False], 'all_pass': all(successes),
                     'roof_color': q['attribute_type'] == 'roof_color',
                     'vehicle_color': q['attribute_type'] == 'vehicle_color'}[criterion]
            if valid:
                selected.append((criterion, q)); seen.add(key); added += 1
            if added == 2:
                break
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = root / 'comparisons/cases'
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for number, (criterion, q) in enumerate(selected):
        key = q['query_sample_id'], q['attribute_type']
        top = [p[key]['ranking'][0] for p in predictions]
        ids = list(dict.fromkeys([q['query_sample_id']] + top + q['positive_sample_ids']))[:8]
        fig, axes = plt.subplots(1, len(ids), figsize=(3 * len(ids), 4))
        if len(ids) == 1:
            axes = [axes]
        for ax, sid in zip(axes, ids):
            row = index[sid]
            ax.imshow(Image.open(row['roi_path']).convert('RGB'))
            title = f"object {row['object_id']} / native {row['roi_pixel_size']}\n"
            title += 'weak positive' if sid in q['positive_sample_ids'] else 'weak negative'
            title += '\nTop1: ' + ', '.join(n.replace('remoteclip_', 'RC ').replace('siglip2_large384', 'Sig') for n, t in zip(names, top) if t == sid)
            ax.set_title(title, fontsize=8, color='green' if sid in q['positive_sample_ids'] else 'darkred')
            ax.axis('off')
        fig.suptitle(criterion + ' / ' + q['map_name'] + '\n' + '\n'.join(textwrap.wrap(q['attribute_phrase'], 95)) + '\nLanguage weak labels; source/cutoff identical across models', fontsize=10)
        fig.tight_layout()
        path = out / f'{number:02d}_{criterion}.png'
        fig.savefig(path, dpi=140)
        plt.close(fig)
        records.append(dict(case=str(path), criterion=criterion, query=q,
                            model_top1=dict(zip(names, top)), displayed_sample_ids=ids))
    (out / 'case_manifest.json').write_text(json.dumps(records, indent=2))
    print(json.dumps({'cases': len(records), 'split': 'val_seen', 'new_model_inference': False}))


if __name__ == '__main__':
    main()
