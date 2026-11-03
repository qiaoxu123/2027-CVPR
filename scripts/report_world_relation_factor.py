"""Posthoc TRAIN report, paired diagnostics and actual cached RGB examples."""
import argparse
from collections import defaultdict
import html
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from graphnav.checkpointing import atomic_json, file_sha256
from graphnav.visual_chain_position import WorldLikelihoodPositionHead
from graphnav.world_relation_factor import WorldRelationFactor
from scripts.train_world_relation_factor import load_data, collate, forward_inputs, evaluate


def summarize(rows, ne_key='ne_m', success_key='success'):
    return dict(frames=len(rows), goal_r1=float(np.mean([r[success_key] for r in rows])),
                goal_ne_m=float(np.mean([r[ne_key] for r in rows])))


def world_grid_extent(world):
    """imshow expects pixel edges, whereas public hypotheses are cell centers."""
    dx = (world[0, 1, 0]-world[0, 0, 0])/2
    dy = (world[1, 0, 1]-world[0, 0, 1])/2
    return [float(world[0, 0, 0]-dx), float(world[0, -1, 0]+dx),
            float(world[0, 0, 1]-dy), float(world[-1, 0, 1]+dy)]


def paired_ci(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['task_key']].append(row)
    differences = np.array([sum(int(r['success'])-int(r['base_success']) for r in values) for values in groups.values()])
    weights = np.array([len(values) for values in groups.values()])
    ids = np.random.default_rng(0).integers(0, len(groups), size=(10000, len(groups)))
    delta = differences[ids].sum(-1)/weights[ids].sum(-1)*100
    return np.quantile(delta, [.025, .975]).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier report')
    status = json.loads((args.suite/'status.json').read_text())
    if status['status'] != 'complete':
        raise ValueError('Completed GPU jobs required')
    torch.set_num_threads(4)
    args.output.mkdir(parents=True)
    results = {}; training = {}; states = {}
    for mode in ('hidden', 'typed'):
        folder = args.suite/(mode+'_s0')
        results[mode] = json.loads((folder/'development.json').read_text())
        training[mode] = json.loads((folder/'training.json').read_text())
        states[mode] = torch.load(folder/'best.pt', map_location='cpu', weights_only=True)
        for filename in ('config.json', 'training.json', 'development.json'):
            shutil.copy2(folder/filename, args.output/(mode+'_'+filename))
    config = states['typed']['config']
    data_args = SimpleNamespace(**{key: Path(value) if key in ('cache', 'fit_labels', 'dev_labels', 'fit_observation_filter') else value
                                  for key, value in config.items()})
    roles = json.loads(Path(config['roles']).read_text())
    catalog = json.loads(Path(config['catalog']).read_text())
    role_vectors = torch.load(config['base_role_vectors'], map_location='cpu', weights_only=True)
    text_vectors = torch.load(config['reference_vectors'], map_location='cpu', weights_only=True)
    fit = load_data(data_args, 'fit', role_vectors, roles, text_vectors, catalog)
    dev = load_data(data_args, 'dev', role_vectors, roles, text_vectors, catalog)
    models = {}
    fit_results = {}
    for mode in ('hidden', 'typed'):
        models[mode] = WorldRelationFactor(WorldLikelihoodPositionHead()).cuda().eval()
        models[mode].load_state_dict(states[mode]['model'])
        fit_results[mode] = evaluate(models[mode], fit, 'cuda', 8, mode)
    typed_rows = results['typed']['results']['real']['rows']
    hidden_rows = results['hidden']['results']['real']['rows']
    methods = {
        'A0': summarize(typed_rows, 'a0_ne_m', 'a0_success'),
        'Frozen V3': summarize(typed_rows, 'base_ne_m', 'base_success'),
        'Hidden': results['hidden']['results']['real']['metrics'],
        'Typed': results['typed']['results']['real']['metrics'],
        'Typed + shuffled references': results['typed']['results']['shuffle-references']['metrics'],
        'Typed + hidden relation types': results['typed']['results']['hidden-relations']['metrics'],
        'Typed + no new map score': results['typed']['results']['no-map-score']['metrics'],
    }
    paired = {}
    for control, value in results['typed']['results'].items():
        paired[control] = dict(goal_positions_changed=sum(a['predicted_world'] != b['predicted_world'] for a, b in zip(typed_rows, value['rows'])),
                              success_changed=sum(a['success'] != b['success'] for a, b in zip(typed_rows, value['rows'])))
    active = [r for r in typed_rows if r['active_references']]
    conditional = dict(active=summarize(active), inactive=summarize([r for r in typed_rows if not r['active_references']]))
    categories = {category: {name: summarize([r for r in rows if r['category'] == category])
                            for name, rows in [('hidden', hidden_rows), ('typed', typed_rows)]}
                  for category in sorted({r['category'] for r in typed_rows})}
    sample = next(row for row in dev if row['reference_valid'].any())
    inputs = forward_inputs(collate([sample]), 'cuda')
    def benchmark(call):
        with torch.inference_mode():
            for _ in range(10): call()
            torch.cuda.synchronize(); started = time.perf_counter()
            for _ in range(100): call()
            torch.cuda.synchronize()
        return (time.perf_counter()-started)*10
    base_ms = benchmark(lambda: models['typed'].base(**{key: inputs[key] for key in models['typed'].BASE_INPUTS}))
    total_ms = benchmark(lambda: models['typed'](**inputs))
    report = dict(methods=methods, paired_controls=paired, conditional=conditional, by_category=categories,
        fit={mode: value['metrics'] for mode, value in fit_results.items()},
        grouped95_delta_r1_pp=paired_ci(typed_rows),
        hidden_vs_typed_positions_changed=sum(a['predicted_world'] != b['predicted_world'] for a, b in zip(typed_rows, hidden_rows)),
        reference_mass_median=float(np.median([r['reference_mass'] for r in active])),
        head_benchmark=dict(base_ms=base_ms, with_factor_ms=total_ms, added_ms=total_ms-base_ms,
                            note='Single-state cached-feature GPU head only, excludes RGB and text encoding.'),
        checkpoints={mode: dict(sha256=results[mode]['checkpoint_sha256'], selected_epoch=results[mode]['selected_epoch']) for mode in results},
        base_checkpoint_sha256=file_sha256(config['base_checkpoint']),
        independent_confirmation_run=False, navigation_run=False,
        decision='No enlargement: typed predicates did not beat hidden predicates; reference RGB shuffle unchanged final Goals.')
    atomic_json(report, args.output/'summary.json')
    atomic_json(status, args.output/'suite_status.json')
    for name in ('smoke', 'resume-smoke', 'hidden', 'typed'):
        shutil.copy2(args.suite/(name+'.log'), args.output/(name+'.log'))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    names = ['A0', 'Frozen V3', 'Hidden', 'Typed', 'Typed + shuffled references']
    axes[0].bar(range(len(names)), [methods[n]['goal_r1']*100 for n in names])
    axes[0].set_ylabel('TRAIN Dev Goal R@1 (%)'); axes[0].set_xticks(range(len(names)), ['A0', 'V3', 'Hidden', 'Typed', 'Shuffle'], rotation=25)
    for mode in training:
        axes[1].plot([e['epoch'] for e in training[mode]['history']], [e['loss'] for e in training[mode]['history']], 'o-', label=mode)
        axes[2].plot([e['epoch'] for e in training[mode]['history']], [e['dev']['goal_ne_m'] for e in training[mode]['history']], 'o-', label=mode)
    axes[1].set_ylabel('TRAIN fitting loss'); axes[2].set_ylabel('TRAIN Dev Goal NE (m)')
    for ax in axes[1:]: ax.set_xlabel('Epoch'); ax.legend()
    fig.tight_layout(); fig.savefig(args.output/'training_comparison.png', dpi=150); plt.close(fig)

    frames = {f['frame_key']: f for f in map(json.loads, (args.observations/'public_frames.jsonl').read_text().splitlines())}
    truths = {f['frame_key']: f for f in map(json.loads, Path(config['dev_labels']).read_text().splitlines())}
    data = {r['frame_key']: r for r in dev}
    selected = []; tasks = set()
    # Include correction, persistent failures, and largest actual Goal changes.
    order = sorted(typed_rows, key=lambda r: (-(r['success'] and not r['base_success']), r['success'], -abs(r['ne_m']-r['base_ne_m']), -r['ne_m']))
    for row in order:
        if row['task_key'] not in tasks:
            selected.append(row); tasks.add(row['task_key'])
        if len(selected) == 6: break
    cases = []; gallery = []
    (args.output/'cases').mkdir()
    for index, row in enumerate(selected):
        frame = frames[row['frame_key']]; record = data[row['frame_key']]
        rgb_path = args.observations/frame['image']
        if file_sha256(rgb_path) != frame['image_sha256']:
            raise ValueError('Actual cached RGB changed')
        x = forward_inputs(collate([record]), 'cuda')
        with torch.inference_mode():
            pred, details = models['typed'](**x, return_details=True)
            base = models['typed'].base(**{key: x[key] for key in models['typed'].BASE_INPUTS})
        world = record['candidate_world'][:784].numpy().reshape(28, 28, 2)
        extent = world_grid_extent(world)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(Image.open(rgb_path), interpolation='nearest'); axes[0].set_title('Actual current RGB (384px)'); axes[0].axis('off')
        truth = truths[row['frame_key']]['target_world'][:2]
        for ax, field, title in zip(axes[1:], (base, pred), ('Frozen V3', 'Typed factor')):
            prob = field[0, :784].softmax(-1).cpu().numpy().reshape(28, 28)
            ax.imshow(prob, origin='lower', extent=extent, cmap='magma')
            footprint = np.array(frame['footprint_world']); footprint = np.vstack((footprint, footprint[0]))
            ax.plot(footprint[:, 0], footprint[:, 1], 'c-', label='Observed footprint')
            ax.scatter(*truth, c='lime', marker='x', label='TRAIN GT point')
            choice = record['candidate_world'][field[0, :784].argmax().item()].numpy()
            ax.scatter(*choice, c='white', marker='+', label='Predicted Goal')
            ax.set_title(title); ax.legend(fontsize=6)
        fig.suptitle(row['frame_key']+' | '+row['category']); fig.tight_layout()
        name = f'cases/{index:02d}_{row["frame_key"]}.png'; fig.savefig(args.output/name, dpi=140); plt.close(fig)
        image_copy = f'cases/{index:02d}_rgb.png'; shutil.copy2(rgb_path, args.output/image_copy)
        case = dict(**row, instruction=frame['instruction'], role=roles[row['task_key']],
                    image_sha256=frame['image_sha256'], figure=name, actual_rgb=image_copy)
        cases.append(case)
        gallery.append(f'<section><h2>{html.escape(row["frame_key"])}</h2><p>{html.escape(frame["instruction"])}</p><a href="{name}"><img width="100%" src="{name}"></a><pre>{html.escape(json.dumps(roles[row["task_key"]]["references"],ensure_ascii=False,indent=2))}</pre></section>')
    atomic_json(cases, args.output/'cases.json')
    (args.output/'index.html').write_text('<meta charset="utf-8"><title>TRAIN关系因子实验</title><style>body{max-width:1200px;margin:24px auto;font:16px sans-serif}pre{white-space:pre-wrap}</style><h1>冻结SigLIP2关系因子TRAIN实验</h1><p>未验证独立收益；不是导航SR。RGB为真实384px观测，GT点只在事后地图显示，不是像素实例框。</p><a href="REPORT_zh.md">中文报告</a><img width="100%" src="training_comparison.png">'+''.join(gallery))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
