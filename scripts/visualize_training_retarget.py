"""Seeded TRAIN-only geometry-core review on actual current RGB, no inference."""
import argparse
import json
from pathlib import Path
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from graphnav.checkpointing import atomic_json, file_sha256
from graphnav.landmark_projection import world_to_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('proposals', 'observations', 'catalog', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Preserve prior review')
    proposals = json.loads((args.proposals/'links_private.json').read_text())
    candidates = [r for r in proposals if r['has_other_instance_core_negatives']]
    random.Random(20261104).shuffle(candidates)
    selected = []; seen_frames = set(); seen_groups = set(); maps = {}
    for proposal in candidates:
        group = (proposal['map_name'], proposal['private_training_instance_id'])
        if proposal['frame_key'] in seen_frames or group in seen_groups or maps.get(group[0], 0) >= 2: continue
        selected.append(proposal); seen_frames.add(proposal['frame_key']); seen_groups.add(group)
        maps[group[0]] = maps.get(group[0], 0)+1
        if len(selected) == 12: break
    frames = {f['frame_key']: f for f in map(json.loads, (args.observations/'public_frames.jsonl').read_text().splitlines())}
    catalog = json.loads(args.catalog.read_text()); rows = []
    args.output.mkdir(parents=True)
    for index, proposal in enumerate(selected):
        frame = frames[proposal['frame_key']]; path = args.observations/frame['image']
        if file_sha256(path) != frame['image_sha256']: raise ValueError('Actual observed RGB changed')
        image = np.array(Image.open(path).convert('RGB'))
        gt = catalog[proposal['map_name']][proposal['private_training_instance_id']]['position'][:2]
        center = world_to_frame(np.array([gt]), frame['footprint_world'], image.shape)[0]
        pixels = np.array([[(i % 12*2+1+.5)*16-.5, (i//12*2+1+.5)*16-.5] for i in proposal['candidate_indices']])
        nearest = pixels[np.linalg.norm(pixels-center, axis=1).argmin()]
        x, y = np.rint(nearest).astype(int); x0, y0 = max(0, x-32), max(0, y-32)
        crop = image[y0:min(384, y+33), x0:min(384, x+33)]
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(image, interpolation='nearest'); axes[0].scatter(*center, marker='x', c='red', s=80)
        axes[0].scatter(pixels[:, 0], pixels[:, 1], facecolors='none', edgecolors='cyan', s=70)
        axes[0].set_title('Actual RGB 384px; red=TRAIN map point')
        axes[1].imshow(crop, interpolation='nearest')
        axes[1].scatter(center[0]-x0, center[1]-y0, marker='x', c='red', s=90)
        axes[1].scatter(pixels[:, 0]-x0, pixels[:, 1]-y0, facecolors='none', edgecolors='cyan', s=90)
        axes[1].set_title('Pixel zoom only; cyan=candidate core')
        for ax in axes: ax.axis('off')
        fig.suptitle(f"{index:02d} {proposal['frame_key']} | geometry proxy, NOT pixel GT", fontsize=9)
        fig.tight_layout(); filename=f'{index:02d}_{proposal["frame_key"]}.png'
        fig.savefig(args.output/filename, dpi=140); plt.close(fig)
        rows.append(dict(**proposal, review_index=index, figure=filename, target_map_point_pixel=center.tolist(),
                         candidate_pixels=pixels.tolist(), image_sha256=frame['image_sha256'],
                         review_status='unreviewed', human_verified=False))
    atomic_json(rows, args.output/'sample_private.json')
    atomic_json(dict(seed=20261104, frames=len(rows), maps=len(maps), private_train_annotations_only=True,
                     images_reencoded=False, input_resolution=[384, 384]), args.output/'summary.json')
    print(json.dumps([dict(index=r['review_index'], frame=r['frame_key'], figure=r['figure']) for r in rows], indent=2))


if __name__ == '__main__': main()
