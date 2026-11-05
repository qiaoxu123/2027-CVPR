"""TRAIN public/label-separated inventory for UnifiedPGVB direct-target training."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import rasterio
from graphnav.direct_target_language import select_direct_relations
from graphnav.checkpointing import file_sha256


def main():
    p = argparse.ArgumentParser()
    for k in ('structured', 'data-root', 'output'):
        p.add_argument('--' + k, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Inventory is immutable; choose a fresh output')
    a.output.mkdir(parents=True)
    catalog = json.loads((a.data_root / 'cityrefer/objects.json').read_text())
    source = a.data_root / 'processed_citynav/citynav_train_seen.json'
    actual = json.loads(source.read_text())
    # Join by public map + original text, never by target identity/coordinates.
    lookup = {}
    for i, r in enumerate(actual):
        for j, text in enumerate(r['descriptions']):
            lookup.setdefault((f"{r['area']}_block_{r['block']}", text), (i, j, r))
    parsed = json.load(gzip.open(a.structured, 'rt'))
    eligible, labels, ledger = [], [], []
    rejection = Counter()
    for r in parsed:
        spec = select_direct_relations(r, catalog.get(r['map_name'], {}))
        reasons = [q['reason'] for q in spec['rejected']]
        if not spec['relations']:
            rejection.update(reasons or ['no_direct_relation'])
            ledger.append(dict(episode=r['episode'], eligible=False, reasons=reasons or ['no_direct_relation']))
            continue
        key = (r['map_name'], r['original_text'])
        if key not in lookup:
            rejection['not_in_revised_TRAIN'] += 1
            continue
        episode, instruction_index, raw = lookup[key]
        path = a.data_root / 'rgbd' / (r['map_name'] + '.tif')
        if not path.exists() or not path.with_suffix('.png').exists():
            rejection['RGB_source_missing'] += 1
            continue
        with rasterio.open(path) as ds:
            b = ds.bounds
            bounds = [b.left, b.bottom, b.right, b.top]
        # Eligibility has ALREADY been decided from text and public geometry.
        # Target labels are a separate file and never serialized in model input.
        target = np.asarray(raw['target_positions'][instruction_index], float)[:2]
        if target.shape != (2,) or not np.isfinite(target).all():
            rejection['training_label_missing'] += 1
            continue
        pose_indices = sorted(set([0, len(raw['trajectory']) // 2, len(raw['trajectory']) - 1]))
        poses = []
        for step in pose_indices:
            t = raw['trajectory'][step]
            yaw = float(np.arctan2(t[4], t[3])) if len(t) == 6 else float(t[3])
            poses.append(dict(step=step, pose=[*t[:3], yaw]))
        # Map-disjoint split fixed before any scores. TRAIN Birmingham is dev.
        subset = 'dev' if raw['area'] == 'birmingham' else 'fit'
        anchors = []
        for rel in spec['relations']:
            lm = rel['landmark']
            anchors.append(dict(**rel, contours=[catalog[r['map_name']][str(oid)]['contour']
                                                for oid in lm['public_ids']]))
        index = len(eligible)
        eligible.append(dict(index=index, episode=episode, original_episode=r['episode'],
                             instruction_index=instruction_index, map_name=r['map_name'],
                             instruction=r['original_text'], subset=subset,
                             world_bounds=bounds, structured=spec, anchors=anchors,
                             observations=poses, target_GT_used=False))
        labels.append(dict(index=index, target_world=target.tolist(),
                           target_ids=[str(x) for x in raw['object_ids']],
                           target_gt_source='revised TRAIN label, loss/evaluation only'))
        ledger.append(dict(episode=r['episode'], eligible=True, index=index, reasons=reasons))
    groups = lambda split: {r['map_name'] + '::' + str(oid) for r in eligible
                            if r['subset'] == split for q in r['anchors']
                            for oid in q['landmark']['public_ids']}
    assert not groups('fit') & groups('dev')
    summary = dict(original_instructions=len(parsed), revised_TRAIN_records=len(actual),
                   qualified_queries=len(eligible),
                   split=dict(Counter(r['subset'] for r in eligible)),
                   target_category=dict(Counter(r['structured']['target']['category'] for r in eligible)),
                   relations=dict(Counter(q['predicate'] for r in eligible for q in r['anchors'])),
                   landmark_type=dict(Counter(q['landmark']['type'] for r in eligible for q in r['anchors'])),
                   rejection=dict(rejection), grouped_public_ID_overlap=0,
                   eligibility_uses_GT=False, pixel_annotations='none; map weak supervision only',
                   cardinal_geography='current engineering convention retained; geographic meaning unverified',
                   source_sha256={str(x): file_sha256(x) for x in
                                  [source, a.structured, a.data_root/'cityrefer/objects.json',
                                   Path('graphnav/direct_target_language.py')]})
    for name, content in [('public_tasks.json', eligible), ('private_labels.json', labels),
                          ('rejection_ledger.json', ledger), ('audit.json', summary)]:
        (a.output / name).write_text(json.dumps(content, ensure_ascii=False, indent=2) + '\n')
    # Fixed seeded textual review sample for the research agent, not pixel GT.
    review = sorted(eligible, key=lambda r: hashlib.sha256(
        (str(42)+r['instruction']).encode()).hexdigest())[:32]
    (a.output/'text_review.json').write_text(json.dumps(review, ensure_ascii=False, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
