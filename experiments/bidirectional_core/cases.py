"""Post-evaluation galleries from exact observed poses; no future-image lookup."""
import argparse
import json
from pathlib import Path
import textwrap

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .cache import CachedCoreDataset
from .data import DEFAULT_DATA
from graphnav.renderer import CityNavObservedViewRenderer


def run(args):
    cv2.setNumThreads(1)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    dataset = CachedCoreDataset(args.cache, condition='E1')
    rows = {dataset[i]['metadata']['sample_id']:dataset[i] for i in range(len(dataset))}
    predictions = {}
    for condition in ('E0','E1','E2'):
        path = Path(args.predictions)/f'{condition}_predictions.jsonl'
        predictions[condition] = {p['sample_id']:p for p in map(json.loads, path.read_text().splitlines())}
    available = set.intersection(*(set(p) for p in predictions.values()))
    groups = {}
    for sid in sorted(available):
        result = predictions['E2'][sid]
        if result.get('real_correct') is None:
            continue
        key = (rows[sid]['metadata']['direction'], bool(result['real_correct']))
        groups.setdefault(key, []).append(sid)
    selected = []
    for key in sorted(groups):
        selected.extend(groups[key][:3])
    renderer = CityNavObservedViewRenderer(Path(args.data_root)/'rgbd', output/'environment_private', image_size=384)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    saved = []
    for index, sid in enumerate(selected):
        row = rows[sid]; metadata = row['metadata']; public = row['public']
        pose = np.asarray(metadata['actual_camera_pose'])
        ground = float(public['ground_level'])
        if pose[2] <= ground:
            raise ValueError('Unknown nonpositive-height frame cannot support visual gallery')
        observation = renderer.current_rgbd(metadata['map_name'], pose, ground)
        rgb = cv2.cvtColor(observation['bgr'], cv2.COLOR_BGR2RGB)
        regions = metadata['roi_objects']
        positive = row['labels']['positive_roi_mask'].numpy().astype(bool)
        judged = row['labels']['semantic_valid_mask'].numpy().astype(bool)
        chosen = [i for i in range(len(regions)) if judged[i]]
        chosen.sort(key=lambda i:(not positive[i], regions[i]['object_id']))
        chosen = chosen[:8]
        canvas = Image.new('RGB', (1100, 360+310*((len(chosen)+3)//4)), 'white')
        draw = ImageDraw.Draw(canvas)
        title = f"{metadata['map_name']} | {metadata['direction']} | actual step {metadata['cutoff_step']}"
        draw.text((20, 12), title, font=font, fill='black')
        draw.multiline_text((420, 48), '\n'.join(textwrap.wrap(metadata['instruction'], 65)), font=font, fill='black', spacing=6)
        full = Image.fromarray(rgb); full.thumbnail((300, 300)); canvas.paste(full, (20, 48))
        for j, condition in enumerate(('E0','E1','E2')):
            prediction = predictions[condition][sid]
            caption = f"{condition}: real R@1 {prediction['real_correct']}; swap R@1 {prediction.get('swapped_correct')}"
            draw.text((420, 170+25*j), caption, font=font, fill='black')
        draw.multiline_text((420, 255), 'Colors/shapes: unverified language weak labels.\nPolygons: geometric weak proposals.\nDisplay enlargement adds no native source detail.', font=small, fill='black', spacing=5)
        for j, i in enumerate(chosen):
            r = regions[i]; x0,y0,x1,y1 = map(int, r['bbox'])
            crop = Image.fromarray(rgb[y0:y1, x0:x1]); crop.thumbnail((240, 200))
            x, y = 20+(j%4)*270, 370+(j//4)*310
            canvas.paste(crop, (x, y))
            score_text = ' / '.join(f"{c}: {float(predictions[c][sid]['roi_scores'][i]):.3f}" for c in ('E0','E1','E2'))
            detail = f"ID {r['object_id']} {r['object_type']}\nNative {r['pixel_size']} | {'weak positive' if positive[i] else 'judged weak negative'}\n{score_text}"
            draw.multiline_text((x,y+210), detail, font=small, fill='black', spacing=5)
        name = f'case_{index:02}.png'; canvas.save(output/name)
        saved.append(dict(sample_id=sid, path=name, annotation_status='geometry/language weak; not verified color segmentation'))
    (output/'manifest.json').write_text(json.dumps(saved, indent=2))
    print(json.dumps(dict(saved_cases=len(saved), output=str(output))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--data-root', default=DEFAULT_DATA)
    run(parser.parse_args())


if __name__=='__main__': main()
