"""Read-only DEV audit; hypothetical public road grouping never changes a cache."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from shapely.geometry import Polygon
from transformers import AutoTokenizer

from experiments.a2_waldo_landmark_belief.data_protocol import (
    _normalize, resolve_public_references,
)
from experiments.a2_waldo_landmark_belief.mentions import landmark_mention_spans
from experiments.a2_waldo_landmark_belief.paths import DATA, sha
from experiments.a2_waldo_landmark_belief.relation_v3_data import (
    RelationV3Views, build_training_row, read_training_source,
)
from experiments.a2_waldo_landmark_belief.relation_v3_fusion import spatial_utility_labels


def inspect_public_matches(matches):
    """Describe topology, without selecting a destination or resolving any link."""
    polygons = [Polygon(x['contour']) for x in matches]
    valid = all(not p.is_empty and p.is_valid and p.area > 0 for p in polygons)
    components = []
    if valid:
        remaining = set(range(len(polygons)))
        while remaining:
            component = {remaining.pop()}
            while True:
                linked = {i for i in remaining
                          if any(polygons[i].intersects(polygons[j]) for j in component)}
                if not linked:
                    break
                component.update(linked)
                remaining.difference_update(linked)
            components.append(sorted(component))
    kinds = {str(x.get('object_type')) for x in matches}
    return dict(
        object_ids=[x['id'] for x in matches], types=sorted(kinds),
        bounds=[list(p.bounds) for p in polygons], valid_geometry=valid,
        connected_components=components,
        hypothetical_connected_road_group=bool(
            len(matches) > 1 and valid and len(components) == 1
            and 'TrafficRoad' in kinds and kinds <= {'TrafficRoad', 'Footpath'}),
    )


def main():
    torch.set_num_threads(2)
    root = Path('runs/a2_waldo_landmark_20261027/revision_v3')
    output = root/'remote_migration'
    packet_path = output/'dev_epoch_04.npz'
    assert sha(packet_path) == 'decb54516bd54ec2eab9dfc2e782e9c0c360acb969664a2c53238ae336abc0b7'
    packets = np.load(packet_path)
    dataset = RelationV3Views('dev')
    source = json.loads((root.parent/'source_manifest.json').read_text())
    annotation = Path(source['annotation_path'])
    assert sha(annotation) == source['annotation_sha256']
    records = json.loads(annotation.read_text())
    objects = json.loads((DATA/'cityrefer/objects.json').read_text())
    processed = json.loads((DATA/'cityrefer/processed_descriptions.json').read_text())
    population = {e['index']: e for e in json.loads(
        (root.parent/'revision_v1/confirmation_population.json').read_text())['entries']}
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
    counts = Counter()
    groups = {}
    cases = []
    for index, entry in enumerate(dataset.entries):
        if entry['source_kind'] == 'teacher_snapshot' or packets['before_ne'][index] <= 20:
            continue
        raw = read_training_source(entry, 'dev', dataset.allowed,
                                   dataset.contexts[entry['source_kind']]['signature'])
        target_kind = dataset.target_kinds[(entry['map_name'], entry['object_id'])]
        row = build_training_row(raw, target_kind=target_kind, real=True)
        spatial, network, labels = row['public']['spatial'], row['public']['network'], row['labels']
        if labels['localization_utility'].any():
            continue
        near = (spatial['grid_world']-labels['target_xy']).norm(dim=-1) <= 20
        if not (near & spatial['observed'].bool()).any():
            continue
        compatible = network['box_kind'] == target_kind if target_kind >= 0 else torch.ones_like(network['box_valid'])
        geometry = spatial_utility_labels({k: v[None] for k, v in spatial.items()}, labels['target_xy'][None])[0]
        if not (geometry & compatible & network['box_valid'].bool()).any():
            continue
        assert entry['source_kind'] == 'revision_v1_A0_closedloop'
        record, identity = records[entry['episode']], population[entry['episode']]
        assert (f"{record['area']}_block_{record['block']}", int(record['object_ids'][0]),
                int(record['ann_ids'][0])) == (entry['map_name'], entry['object_id'], identity['description_id'])
        instruction = str(record['descriptions'][0])
        queries = list(dict.fromkeys(str(x) for x in processed[entry['map_name']][str(entry['object_id'])][identity['description_id']].get('landmarks', [])))
        assert len(queries) == identity['reference_queries']
        named = [x for x in objects[entry['map_name']].values() if x.get('name')]
        refs = resolve_public_references(entry['map_name'], queries, named)
        offsets = tokenizer(instruction, truncation=True, max_length=96, return_offsets_mapping=True)['offset_mapping']
        _, linked = landmark_mention_spans(instruction, offsets, refs)
        assert int(linked.sum()) == raw['meta']['valid_references'] == 0
        reason = 'ambiguous_public_names' if refs and not any(x['linked'] for x in refs) else 'name_absent_or_unresolved_in_instruction'
        counts[reason] += 1
        query_keys = []
        if reason == 'ambiguous_public_names':
            for query in queries:
                key = entry['map_name']+'\0'+_normalize(query)
                if key not in groups:
                    matches = [x for x in named if _normalize(x['name']) == _normalize(query)]
                    groups[key] = dict(map_name=entry['map_name'], name=query,
                                       **inspect_public_matches(matches), observation_indices=[], targets=[])
                group = groups[key]
                group['observation_indices'].append(index)
                target = [entry['map_name'], entry['object_id']]
                if target not in group['targets']:
                    group['targets'].append(target)
                query_keys.append(key)
        hypothetical_refs = [dict(name=groups[k]['name'], aliases=[groups[k]['name']],
                                  linked=groups[k]['hypothetical_connected_road_group'])
                             for k in query_keys]
        _, hypothetical_links = landmark_mention_spans(instruction, offsets, hypothetical_refs)
        cases.append(dict(dev_index=index, episode=entry['episode'], target=[entry['map_name'], entry['object_id']],
                          filename=entry['filename'], reason=reason, query_keys=query_keys,
                          hypothetical_public_mention_unblocked=bool(hypothetical_links.any())))
    assert len(cases) == 94 and counts['ambiguous_public_names'] == 86
    assert not torch.cuda.is_initialized()
    applicable = [x for x in cases if x['hypothetical_public_mention_unblocked']]
    result = dict(status='public_topology_and_cluster_size_audit_complete', counts=dict(counts),
                  distinct_targets=len({tuple(x['target']) for x in cases}),
                  ambiguous_distinct_targets=len({tuple(x['target']) for x in cases if x['reason']=='ambiguous_public_names'}),
                  hypothetical_applicable_observations=len(applicable),
                  hypothetical_applicable_targets=len({tuple(x['target']) for x in applicable}),
                  groups=groups, cases=cases, no_gpu_used=True, training_changed=False,
                  benchmark_used=False, calibration_used=False,
                  limitations=['Connected same-name road surfaces are a grouping hypothesis, not verified entity identity.',
                               'Counts describe repeated DEV observations, not independent episodes or proven navigation gains.'])
    path = output/'epoch04_public_topology_audit.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('groups', 'cases')}))
    for group in groups.values():
        print(json.dumps({k: v for k, v in group.items() if k not in ('observation_indices', 'targets')}))


if __name__ == '__main__':
    main()
