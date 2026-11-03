"""Posthoc Goal/region transfer and TRAIN supervision audit, no model updates."""
import argparse
from collections import Counter
import html
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from graphnav.checkpointing import atomic_json, file_sha256
from graphnav.region_auxiliary_loss import RegionAuxiliaryHead
from graphnav.region_semantic_probe import classification_metrics
from graphnav.training_supervision_audit import position_region_counts, car_core_position_counts
from graphnav.visual_chain_position import WorldLikelihoodPositionHead
from scripts.train_local_chain_position import records, batch_inputs
from scripts.train_region_auxiliary_goal import collate, paired_changes
from scripts.report_world_relation_factor import world_grid_extent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('suite', 'observations', 'old-car-core', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Preserve prior report')
    status = json.loads((args.suite/'status.json').read_text())
    if status['status'] != 'complete' or any(s['returncode'] for s in status['stages']):
        raise ValueError('Completed GPU jobs required')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    results = {}; states = {}; models = {}; auxiliary = {}
    for name, folder in [('A', 'goal_only_s0'), ('B', 'goal_region_s0')]:
        directory = args.suite/folder
        results[name] = json.loads((directory/'development.json').read_text())
        states[name] = torch.load(directory/'final.pt', map_location='cpu', weights_only=True)
        models[name] = WorldLikelihoodPositionHead().eval(); models[name].load_state_dict(states[name]['model'], strict=True)
        auxiliary[name] = RegionAuxiliaryHead().eval(); auxiliary[name].load_state_dict(states[name]['region_auxiliary'])
        for filename in ('config.json', 'training.json', 'development.json', 'status.json'):
            shutil.copy2(directory/filename, args.output/(name+'_'+filename))
    config = states['B']['config']
    roles = torch.load(config['role_vectors'], map_location='cpu', weights_only=True)
    masks = torch.load(Path(config['region_labels'])/'loss_labels_private.pt', map_location='cpu', weights_only=True)
    allowed = set(json.loads(Path(config['fit_observation_filter']).read_text())['frame_keys'])
    fit = records(Path(config['cache']), Path(config['fit_labels']), 'fit', roles, 'reference', allowed)
    dev = records(Path(config['cache']), Path(config['dev_labels']), 'dev', roles, 'reference')
    core_proof = json.loads((args.old_car_core.parent/'summary.json').read_text())
    if core_proof['private_labels_sha256'] != file_sha256(args.old_car_core) or core_proof['primary_feature_manifest_sha256'] != file_sha256(Path(config['cache'])/'manifest.json'):
        raise ValueError('old private instance core provenance differs')
    cores = torch.load(args.old_car_core, map_location='cpu', weights_only=True)
    mixture = {}; core_rows = []
    for subset, data in [('fit', fit), ('dev', dev)]:
        categories = {category: Counter() for category in ('Building', 'Parking', 'Ground')}
        examples = []
        for row in data:
            distance = (row['candidate_world'][784:]-row['target_world']).norm(dim=-1).numpy()
            label = masks[row['frame_key']].numpy(); valid = row['local_valid'].numpy()
            counts = position_region_counts(distance, valid, label, row['category'])
            if counts['applicable']:
                c = categories[row['category']]; c.update({k: v for k, v in counts.items() if k != 'applicable'})
                c['frames'] += 1; c['frames_with_other_region_positive'] += int(counts['other_region'] > 0)
                if counts['other_region'] > 0:
                    examples.append(dict(frame_key=row['frame_key'], task_key=row['task_key'], category=row['category'], **counts))
            if subset == 'fit' and row['category'] == 'Car' and row['frame_key'] in cores:
                count = car_core_position_counts(distance, valid, cores[row['frame_key']].numpy())
                core_rows.append(dict(frame_key=row['frame_key'], **count))
        mixture[subset] = dict(by_target_category={key: dict(value) for key, value in categories.items()}, examples=examples)
    core_summary = dict(frames_with_existing_core_labels=len(core_rows),
        frames_with_other_car_in_position_positives=sum(r['other_car_core_points'] > 0 for r in core_rows),
        target_core_points=sum(r['target_core_points'] for r in core_rows),
        other_car_core_points=sum(r['other_car_core_points'] for r in core_rows),
        position_positive_points=sum(r['position_positive_points'] for r in core_rows),
        frames_with_target_core=sum(r['target_core_points'] > 0 for r in core_rows),
        note='Registered TRAIN world-position cores only; not verified RGB identities or visibility.')
    truths = []; class_predictions = []
    with torch.inference_mode():
        for row in dev:
            label = masks[row['frame_key']].numpy(); known = (label >= 0) & row['local_valid'].numpy()
            logits = auxiliary['B'](image_features=row['image_features'][None], projection=models['B'].image)
            truths.extend(label[known]); class_predictions.extend(logits[0].argmax(-1).numpy()[known])
    methods = {'frozen_V3': results['B']['frozen_baseline']['metrics']}
    for name in results:
        for control in ('real', 'zero', 'shuffle', 'no_geometry', 'full_text'):
            methods[name+'_'+control] = results[name]['results'][control]['metrics']
    summary = dict(methods=methods, B_vs_A=paired_changes(results['B']['results']['real'], results['A']['results']['real']),
        B_auxiliary_region_proxy=classification_metrics(truths, class_predictions),
        region_supervision_mixture=mixture, car_core_supervision=core_summary,
        checkpoint_sha256={name: value['checkpoint_sha256'] for name, value in results.items()},
        navigation_run=False, independent_confirmation_run=False,
        decision='No promotion: both retrained models lose three correct states versus frozen V3; no consistent visual-coordinate gain.')
    atomic_json(summary, args.output/'summary.json')
    atomic_json(core_rows, args.output/'car_core_rows_private_diagnostic.json')
    atomic_json(status, args.output/'suite_status.json')
    for name in ('smoke', 'resume-smoke', 'goal-only', 'goal-region'):
        shutil.copy2(args.suite/(name+'.log'), args.output/(name+'.log'))
    initial = torch.load(config['warm_start'], map_location='cpu', weights_only=True)
    models['frozen_V3'] = WorldLikelihoodPositionHead().eval(); models['frozen_V3'].load_state_dict(initial['model'])
    reference = {r['frame_key']: r for r in results['B']['frozen_baseline']['rows']}
    ar = {r['frame_key']: r for r in results['A']['results']['real']['rows']}
    br = {r['frame_key']: r for r in results['B']['results']['real']['rows']}
    selected = []; seen = set()
    for key in br:
        if reference[key]['success'] and (not ar[key]['success'] or not br[key]['success']):
            selected.append(key); seen.add(br[key]['task_key'])
    candidates = sorted(br, key=lambda k:ar[k]['ne_m']-br[k]['ne_m'], reverse=True)
    for key in candidates:
        if len(selected) >= 6: break
        if br[key]['task_key'] not in seen:
            selected.append(key); seen.add(br[key]['task_key'])
    frames = {r['frame_key']: r for r in map(json.loads, (args.observations/'public_frames.jsonl').read_text().splitlines())}
    feature = {r['frame_key']: r for r in dev}; cases = []; gallery = []
    (args.output/'cases').mkdir()
    for index, key in enumerate(selected):
        row = feature[key]; frame = frames[key]; path = args.observations/frame['image']
        if file_sha256(path) != frame['image_sha256']: raise ValueError('Actual current RGB changed')
        public = batch_inputs(collate([row]), 'cpu')
        world = row['candidate_world'][:784].numpy().reshape(28, 28, 2); extent = world_grid_extent(world)
        fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
        axes[0].imshow(Image.open(path), interpolation='nearest'); axes[0].axis('off'); axes[0].set_title('Actual current RGB (384px)')
        for ax, name in zip(axes[1:], ('frozen_V3', 'A', 'B')):
            with torch.no_grad(): logits = models[name](**public)[0, :784]
            ax.imshow(logits.softmax(-1).reshape(28, 28).numpy(), origin='lower', extent=extent, cmap='magma')
            polygon = np.array(frame['footprint_world']); polygon = np.vstack((polygon, polygon[0]))
            ax.plot(polygon[:, 0], polygon[:, 1], 'c-', linewidth=.8)
            ax.scatter(*row['target_world'].numpy(), c='lime', marker='x', label='TRAIN GT point')
            point = row['candidate_world'][logits.argmax()].numpy(); ax.scatter(*point, c='white', marker='+', label='Goal')
            ax.set_title(name); ax.legend(fontsize=6)
        fig.suptitle(key+' | offline observation; GT point only, not pixel instance box', fontsize=9)
        fig.tight_layout(); filename=f'cases/{index:02d}_{key}.png'; fig.savefig(args.output/filename, dpi=140); plt.close(fig)
        case = dict(frame_key=key, instruction=frame['instruction'], category=row['category'], image_sha256=frame['image_sha256'],
            frozen=reference[key], A=ar[key], B=br[key], figure=filename)
        cases.append(case); gallery.append(f'<section><h2>{html.escape(key)}</h2><p>{html.escape(frame["instruction"])}</p><a href="{filename}"><img width="100%" src="{filename}"></a></section>')
    atomic_json(cases, args.output/'cases.json')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for name in ('A', 'B'):
        history = states[name]['history']; axes[0].plot([r['epoch'] for r in history], [r['region_loss'] for r in history], 'o-', label=name)
    axes[0].set_title('Training-only region loss'); axes[0].legend()
    keys = ('frozen_V3', 'A_real', 'B_real', 'B_shuffle')
    axes[1].bar(range(4), [methods[k]['goal_r1']*100 for k in keys]); axes[1].set_ylabel('TRAIN Dev Goal R@1 (%)')
    axes[2].bar(range(4), [methods[k]['goal_ne_m'] for k in keys]); axes[2].set_ylabel('TRAIN Dev Goal NE (m)')
    for ax in axes[1:]: ax.set_xticks(range(4), ['Frozen', 'A', 'B', 'B Shuffle'])
    fig.tight_layout(); fig.savefig(args.output/'comparison.png', dpi=150); plt.close(fig)
    (args.output/'index.html').write_text('<meta charset="utf-8"><title>区域辅助Goal训练</title><style>body{max-width:1500px;margin:24px auto;font:16px sans-serif}</style><h1>区域辅助Goal训练：负结果</h1><p>全部为TRAIN离线；原权重不变，无新SR。GT只在事后地图显示。</p><a href="REPORT_zh.md">中文报告</a><img width="90%" src="comparison.png">'+''.join(gallery))
    print(json.dumps(dict(B_vs_A=summary['B_vs_A'], region_proxy=summary['B_auxiliary_region_proxy'], car_core=core_summary,
                         region_mixture={s:v['by_target_category'] for s,v in mixture.items()}), ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
