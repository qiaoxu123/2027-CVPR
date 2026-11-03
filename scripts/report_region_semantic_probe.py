"""Frozen probe results, height strata, and real RGB failures; no retraining."""
import argparse
import html
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from graphnav.checkpointing import atomic_json, file_sha256
from graphnav.region_semantic_probe import RegionSemanticProbe


def merge_metrics(rows, method):
    confusion = sum((np.array(r['methods'][method]['confusion']) for r in rows), np.zeros((3, 3), dtype=np.int64))
    support = confusion.sum(-1); diagonal = confusion.diagonal()
    recall = np.divide(diagonal, support, out=np.zeros(3), where=support > 0)
    f1 = np.divide(2*diagonal, support+confusion.sum(0), out=np.zeros(3), where=support+confusion.sum(0) > 0)
    return dict(frames=len(rows), patches=int(support.sum()), accuracy=float(diagonal.sum()/max(1, support.sum())),
        balanced_accuracy=float(recall[support > 0].mean()) if (support > 0).any() else None,
        macro_f1=float(f1.mean()), class_recall=recall.tolist(), class_support=support.tolist(), confusion=confusion.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('suite', 'observations', 'public-filter', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Preserve prior report')
    status = json.loads((args.suite/'status.json').read_text())
    if status['status'] != 'complete' or any(r['returncode'] for r in status['stages']):
        raise ValueError('Actual completed GPU runs required')
    args.output.mkdir(parents=True)
    pilot = args.suite/'pilot'
    evaluation = json.loads((pilot/'evaluation.json').read_text())
    allowed = set(json.loads(args.public_filter.read_text())['frame_keys'])
    rows = evaluation['results']['dev']['rows']
    stratified = {}
    for name, values in [('height_envelope_or_initial', [r for r in rows if r['frame_key'] in allowed]),
                          ('noninitial_outside_height_envelope', [r for r in rows if r['frame_key'] not in allowed])]:
        stratified[name] = {method: merge_metrics(values, method) for method in evaluation['results']['dev']['metrics']}
    summary = dict(checkpoint_sha256=evaluation['checkpoint_sha256'],
        full_development=evaluation['results']['dev']['metrics'],
        multi_class_frames=evaluation['results']['dev']['multi_class_frames'],
        multi_class_metrics=evaluation['results']['dev']['multi_class_metrics'],
        shuffle_predictions_changed=evaluation['results']['dev']['shuffle_predictions_changed'],
        height_strata_posthoc=stratified, selection='fixed third epoch; no development tuning',
        warning='Geometry-proxy land-use diagnostic only, not pixel truth, Goal accuracy or SR')
    for name in ('config.json', 'training.json', 'evaluation.json', 'status.json'):
        shutil.copy2(pilot/name, args.output/name)
    for name in ('smoke', 'resume-smoke', 'pilot'):
        shutil.copy2(args.suite/(name+'.log'), args.output/(name+'.log'))
    atomic_json(summary, args.output/'summary.json')
    atomic_json(status, args.output/'suite_status.json')
    frames = {r['frame_key']: r for r in map(json.loads, (args.observations/'public_frames.jsonl').read_text().splitlines())}
    labels_root = Path(json.loads((pilot/'config.json').read_text())['config']['labels'])
    labels = torch.load(labels_root/'loss_labels_private.pt', map_location='cpu', weights_only=True)
    cache = Path(json.loads((pilot/'config.json').read_text())['config']['cache'])
    records = json.loads((labels_root/'records.json').read_text())
    indexes = {r['frame_key']: r for r in records}
    state = torch.load(pilot/'final.pt', map_location='cpu', weights_only=True)
    model = RegionSemanticProbe().eval(); model.load_state_dict(state['model'])
    native_path = Path(state['config']['native_text'])
    native = F.normalize(torch.load(native_path, map_location='cpu', weights_only=True).float(), dim=-1)
    # Four poor Parking observations (not selected by availability/visibility)
    # and two mixed-category examples. Distinct task groups, posthoc only.
    selected = []; used = set()
    parking = sorted((r for r in rows if r['methods']['trained']['class_support'][1] >= 10), key=lambda r:r['methods']['trained']['class_recall'][1])
    for row in parking:
        if row['task_key'] in used: continue
        selected.append(row); used.add(row['task_key'])
        if len(selected) == 4: break
    mixed = sorted((r for r in rows if r['multi_class']), key=lambda r:r['methods']['trained']['accuracy'], reverse=True)
    for row in mixed:
        if row['task_key'] in used: continue
        selected.append(row); used.add(row['task_key'])
        if len(selected) == 6: break
    colors = ('#4799ff', '#ffb347', '#36d174'); cases = []; gallery = []
    (args.output/'cases').mkdir()
    for index, row in enumerate(selected):
        frame = frames[row['frame_key']]; rgb_path = args.observations/frame['image']
        if file_sha256(rgb_path) != frame['image_sha256']: raise ValueError('Actual RGB changed')
        chunk = indexes[row['frame_key']]['feature_chunk']
        feature = next(r for r in torch.load(cache/chunk, map_location='cpu', weights_only=True)['records'] if r['frame_key'] == row['frame_key'])
        image = np.array(Image.open(rgb_path).convert('RGB')); label = labels[row['frame_key']].numpy()
        with torch.no_grad():
            prediction = model(image_features=feature['image_features']).argmax(-1).numpy()
            native_pred = (F.normalize(feature['image_features'].float(), dim=-1).mean(-2)@native.T).argmax(-1).numpy()
        fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
        for ax in axes: ax.imshow(image, interpolation='nearest'); ax.axis('off')
        axes[0].set_title('Actual current RGB (384px)')
        for ax, target, title in zip(axes[1:], (label, native_pred, prediction),
                                    ('TRAIN contour proxy', 'Native text match', 'Trained visual probe')):
            for i, value in enumerate(target):
                if label[i] < 0: continue
                x, y = (i % 12)*2+1, (i//12)*2+1
                ax.add_patch(Rectangle((x*16, y*16), 15, 15, facecolor=colors[value], alpha=.55, linewidth=.3, edgecolor='white'))
            ax.set_title(title)
        fig.suptitle(row['frame_key']+' | Blue=Building, Orange=Parking, Green=Ground | only proxy-evaluated patches shown', fontsize=9)
        fig.tight_layout(); filename=f'cases/{index:02d}_{row["frame_key"]}.png'
        fig.savefig(args.output/filename, dpi=140); plt.close(fig)
        case = dict(**row, figure=filename, instruction=frame['instruction'], camera_pose=frame['camera_pose'],
                    height_envelope_or_initial=row['frame_key'] in allowed, rgb_sha256=frame['image_sha256'])
        cases.append(case)
        gallery.append(f'<section><h2>{html.escape(row["frame_key"])}</h2><p>{html.escape(frame["instruction"])}</p><a href="{filename}"><img width="100%" src="{filename}"></a></section>')
    atomic_json(cases, args.output/'cases.json')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    methods = ['class-prior', 'native-text', 'trained', 'trained-shuffle']
    axes[0].bar(range(4), [summary['full_development'][m]['balanced_accuracy']*100 for m in methods]); axes[0].set_xticks(range(4), ['Prior', 'Native', 'Trained', 'Shuffle'])
    axes[0].set_ylabel('TRAIN Dev proxy balanced accuracy (%)')
    cm = np.array(summary['full_development']['trained']['confusion'])
    axes[1].imshow(cm, cmap='Blues'); axes[1].set_xticks(range(3), ['Building', 'Parking', 'Ground']); axes[1].set_yticks(range(3), ['Building', 'Parking', 'Ground'])
    for i in range(3):
        for j in range(3): axes[1].text(j, i, str(cm[i,j]), ha='center', va='center')
    axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Geometry proxy')
    fig.tight_layout(); fig.savefig(args.output/'comparison.png', dpi=150); plt.close(fig)
    (args.output/'index.html').write_text('<meta charset="utf-8"><title>视觉区域语义诊断</title><style>body{max-width:1500px;margin:24px auto;font:16px sans-serif}</style><h1>冻结SigLIP2区域语义诊断</h1><p>全部为TRAIN；Contour只是弱代理，非像素实例GT。本轮没有导航SR。</p><a href="REPORT_zh.md">中文报告</a><img width="80%" src="comparison.png">'+''.join(gallery))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
