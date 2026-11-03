"""Audit literal public-name alternatives without selecting a target instance.

Every homonymous polygon remains a separate alternative. No union polygon,
private target metadata, nearest-target selection, or model changes are used.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import torch
from graphnav.checkpointing import atomic_json, file_sha256
from graphnav.landmark_alias import canonical_name, resolve_references
from graphnav.landmark_projection import project_landmark
from graphnav.public_reference_supervision import patch_world_windows, reference_region_labels
from scripts.analyze_instruction_landmark_association import PublicNames


def public_group(mention, objects):
    """Keep all literal matches; classification does not establish identity."""
    ids = sorted(set(mention['public_ids']))
    members = [objects[i] for i in ids]
    names = {canonical_name(o['name']) for o in members}
    types = {o['object_type'] for o in members}
    kind = ('single_polygon' if len(ids) == 1 else
            'homonymous_road_polygons' if len(names) == 1 and types == {'TrafficRoad'} else
            'homonymous_other_polygons' if len(names) == 1 and len(types) == 1 else
            'name_or_type_alternatives')
    # Reuse production binding, but never silently truncate an audit group.
    bound = resolve_references(mention['public_names'], objects,
                              max_landmarks=max(1, len(objects)), match_mode='exact')
    return dict(kind=kind, public_ids=ids, exact_binding_ids=sorted(x['id'] for x in bound),
                names=sorted(names), types=sorted(types))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'observations', 'catalog', 'filter', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    manifest = json.loads((a.cache/'manifest.json').read_text())
    assert manifest['complete'] and manifest['split'] == 'train_seen' and manifest['camera_yaw_rad'] == 0
    assert file_sha256(a.catalog) == manifest['public_catalog_sha256']
    frames_path = a.observations/'public_frames.jsonl'
    om = json.loads((a.observations/'manifest.json').read_text())
    assert file_sha256(frames_path) == om['public_frames.jsonl_sha256']
    frames = {x['frame_key']: x for x in map(json.loads, frames_path.read_text().splitlines())}
    allowed = set(json.loads(a.filter.read_text())['frame_keys'])
    catalog = json.loads(a.catalog.read_text()); names = PublicNames(catalog)
    rows = []; counts = Counter(); groups_cache = {}
    for chunk in manifest['chunks']:
        path = a.cache/chunk['file']
        assert file_sha256(path) == chunk['sha256']
        for f in torch.load(path, map_location='cpu', weights_only=True)['records']:
            subset = f['subset']
            if subset not in ('fit', 'dev') or (subset == 'fit' and f['frame_key'] not in allowed):
                continue
            frame = frames[f['frame_key']]; key = (frame['map_name'], frame['instruction'])
            if key not in groups_cache:
                groups_cache[key] = [public_group(m, catalog[key[0]]) for m in names.match(*key)]
            windows = patch_world_windows(frame['footprint_world'])
            counts[subset+'_frames'] += 1
            for group in groups_cache[key]:
                members = []
                for oid in group['public_ids']:
                    obj = catalog[key[0]][oid]
                    landmark = {k: obj[k] for k in ('name', 'object_type', 'contour')}
                    landmark['id'] = oid
                    small = project_landmark(landmark, frame['footprint_world'], (384,384), min_visible=0, min_pixels=32)
                    strict = project_landmark(landmark, frame['footprint_world'], (384,384), min_visible=.9, min_pixels=256)
                    positive = []
                    for j in range(3):
                        valid = f['local_valid'].numpy() & (f['quality'][:,j].numpy() >= .9)
                        labels, _ = reference_region_labels(windows[:,j], obj['contour'], valid)
                        positive.append(bool((labels == 1).any() and (labels == 0).any()))
                    members.append(dict(public_id=oid, eligible_patch_scales=positive,
                        projected_32px=small is not None, projected_strict=strict is not None,
                        projected_fraction=small.visible_fraction if small else 0,
                        projected_pixels=small.visible_pixels if small else 0))
                rows.append(dict(frame_key=f['frame_key'], subset=subset, map_name=key[0],
                    group=group, members=members, visibility='unknown', pixel_gt=False))
    summary = dict(counts=dict(counts), by_subset={}, private_target_labels_read=False,
                   normal_inference_modified=False, warning='Projection support only; not RGB visibility or instance gold.')
    for subset in ('fit','dev'):
        rr = [r for r in rows if r['subset'] == subset]; out = {}
        for mode in ('old_single_only','all_public_alternatives','multiple_only'):
            selected = [r for r in rr if mode == 'all_public_alternatives' or
                (len(r['members']) == 1 if mode == 'old_single_only' else len(r['members']) > 1)]
            out[mode] = dict(name_frame_groups=len(selected), distinct_frames=len({r['frame_key'] for r in selected}),
                group_kinds=dict(Counter(r['group']['kind'] for r in selected)))
            for metric in ('projected_32px','projected_strict',0,1,2):
                supported = [r for r in selected if any(m['eligible_patch_scales'][metric] if isinstance(metric,int) else m[metric] for m in r['members'])]
                out[mode][str(metric)] = dict(groups=len(supported), distinct_frames=len({r['frame_key'] for r in supported}))
        summary['by_subset'][subset] = out
    summary['input_sha256'] = {k:file_sha256(v) for k,v in dict(catalog=a.catalog, frames=frames_path, filter=a.filter, manifest=a.cache/'manifest.json', producer=Path(__file__)).items()}
    a.output.mkdir(parents=True)
    atomic_json(rows,a.output/'records.json'); atomic_json(summary,a.output/'summary.json')
    print(json.dumps(summary,indent=2))

if __name__ == '__main__':
    main()
