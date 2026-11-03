"""Inspect seeded TRAIN-fit proxy labels on actual cached 384px RGB."""
import argparse
import json
from pathlib import Path
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
import torch

from graphnav.checkpointing import atomic_json, file_sha256
from graphnav.region_semantic_supervision import REGION_CLASSES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('labels', 'observations', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Preserve inspection')
    args.output.mkdir(parents=True)
    records = json.loads((args.labels/'records.json').read_text())
    labels = torch.load(args.labels/'loss_labels_private.pt', map_location='cpu', weights_only=True)
    frames = {r['frame_key']: r for r in map(json.loads, (args.observations/'public_frames.jsonl').read_text().splitlines())}
    rng = random.Random(20261104); selected = []; used = set()
    for category in REGION_CLASSES:
        candidates = [r for r in records if r['subset'] == 'fit' and r['by_class'][category] >= 8 and r['task_key'] not in used]
        rng.shuffle(candidates)
        for row in candidates:
            if row['task_key'] in used: continue
            selected.append(row); used.add(row['task_key'])
            if len(selected) % 2 == 0: break
    result = []; colors = ('#4799ff', '#ffb347', '#36d174')
    for index, row in enumerate(selected):
        frame = frames[row['frame_key']]; image_path = args.observations/frame['image']
        if file_sha256(image_path) != frame['image_sha256']: raise ValueError('RGB changed')
        image = np.array(Image.open(image_path).convert('RGB')); label = labels[row['frame_key']].numpy()
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
        for ax in axes: ax.imshow(image, interpolation='nearest'); ax.axis('off')
        for i, value in enumerate(label):
            if value < 0: continue
            x, y = (i % 12)*2+1, (i//12)*2+1
            axes[1].add_patch(Rectangle((x*16, y*16), 15, 15, facecolor=colors[value], edgecolor='white', alpha=.55, linewidth=.4))
        axes[0].set_title('Actual current RGB (384px)'); axes[1].set_title('Geometry proxies, NOT pixel GT')
        fig.suptitle(row['frame_key']+' | Blue=Building, Orange=Parking, Green=Ground', fontsize=9)
        fig.tight_layout(); filename=f'{index:02d}_{row["frame_key"]}.png'
        fig.savefig(args.output/filename, dpi=140); plt.close(fig)
        result.append(dict(**row, figure=filename, instruction=frame['instruction'], image_sha256=frame['image_sha256']))
    atomic_json(result, args.output/'cases.json')
    print(json.dumps([dict(frame_key=r['frame_key'], by_class=r['by_class']) for r in result], indent=2))


if __name__ == '__main__': main()
