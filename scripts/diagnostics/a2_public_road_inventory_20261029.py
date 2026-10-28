"""TRAIN/DEV public-name inventory; never deserialize calibration observations."""
import json
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from experiments.a2_waldo_landmark_belief.data_protocol import resolve_public_references
from experiments.a2_waldo_landmark_belief.public_road_references import resolve_road_references
from experiments.a2_waldo_landmark_belief.mentions import landmark_mention_spans
from experiments.a2_waldo_landmark_belief.paths import DATA, sha


def main():
    base = Path('runs/a2_waldo_landmark_20261027')
    roles_path = base/'revision_v3/roles.json'
    roles = json.loads(roles_path.read_text())
    source = json.loads((base/'source_manifest.json').read_text())
    annotation = Path(source['annotation_path'])
    if sha(annotation) != source['annotation_sha256']:
        raise ValueError('Source annotation changed')
    records = json.loads(annotation.read_text())
    objects_path = DATA/'cityrefer/objects.json'
    processed_path = DATA/'cityrefer/processed_descriptions.json'
    objects = json.loads(objects_path.read_text())
    processed = json.loads(processed_path.read_text())
    populations = {'teacher_snapshot': {x['index']: x for x in source['entries']}}
    for version in ('revision_v1', 'revision_v2'):
        manifest = json.loads((base/version/'confirmation_population.json').read_text())
        populations[version+'_A0_closedloop'] = {x['index']: x for x in manifest['entries']}
    named = {m: [v for v in values.values() if v.get('name')] for m, values in objects.items()}
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
    allowed = {role: {(x['map_name'], x['object_id']) for x in values}
               for role, values in roles['target_roles'].items()}
    counters, targets, cache = defaultdict(Counter), defaultdict(set), {}
    loss_examples = []
    for entry in roles['rows']:
        role = entry['role']
        if role not in ('train', 'dev'):
            continue
        target = (entry['map_name'], entry['object_id'])
        if target not in allowed[role] or target in allowed['calibration_fit'] | allowed['calibration_verify']:
            raise ValueError('Forbidden or foreign TRAIN/DEV identity')
        key = (entry['source_kind'], entry['episode'])
        if key not in cache:
            identity = populations[entry['source_kind']][entry['episode']]
            record = records[entry['episode']]
            actual = (f"{record['area']}_block_{record['block']}", int(record['object_ids'][0]), int(record['ann_ids'][0]))
            if actual != (*target, identity['description_id']):
                raise ValueError('Instruction identity mismatch')
            instruction = str(record['descriptions'][0])
            queries = processed[target[0]][str(target[1])][identity['description_id']].get('landmarks', [])
            old = resolve_public_references(target[0], queries, named[target[0]])
            new = resolve_road_references(target[0], queries, named[target[0]])
            offsets = tokenizer(instruction, truncation=True, max_length=96, return_offsets_mapping=True)['offset_mapping']
            _, old_links = landmark_mention_spans(instruction, offsets, old)
            _, new_links = landmark_mention_spans(instruction, offsets, new)
            if old_links.any() and not new_links.any():
                loss_examples.append(dict(source_kind=entry['source_kind'], episode=entry['episode'],
                    instruction=instruction, queries=queries,
                    old_references=[dict(name=r['name'], linked=bool(r['linked']),
                                         valid_mention=bool(old_links[i])) for i,r in enumerate(old)],
                    new_references=[dict(name=r['name'], linked=bool(r['linked']),
                                         valid_mention=bool(new_links[i]), grouped='region' in r)
                                    for i,r in enumerate(new)]))
            cache[key] = dict(old=int(old_links.sum()), new=int(new_links.sum()),
                              grouped=sum('region' in ref for ref in new))
        result = cache[key]
        for bucket in (role, role+'/'+entry['source_kind']):
            counts = counters[bucket]
            counts['observations'] += 1
            counts['old_any_literal_reference'] += result['old'] > 0
            counts['new_any_literal_reference'] += result['new'] > 0
            counts['recovered_from_zero'] += result['old'] == 0 and result['new'] > 0
            counts['lost_all_references'] += result['old'] > 0 and result['new'] == 0
            counts['has_public_road_group'] += result['grouped'] > 0
            if result['old'] == 0 and result['new'] > 0:
                targets[bucket].add(target)
    if counters['train']['observations'] != 33440 or counters['dev']['observations'] != 5647:
        raise ValueError('Incomplete TRAIN/DEV inventory')
    report = dict(status='public_reference_inventory_complete', roles_sha256=sha(roles_path),
                  annotation_sha256=sha(annotation), public_objects_sha256=sha(objects_path),
                  processed_descriptions_sha256=sha(processed_path), unique_source_episodes=len(cache),
                  counts={key: dict(value, recovered_distinct_targets=len(targets[key])) for key,value in counters.items()},
                  lost_reference_source_episode_examples=loss_examples,
                  calibration_observations_opened=False, benchmark_used=False, model_run=False,
                  limitations=['Public literal-reference availability only; no candidate coverage or navigation gain measured.',
                               'Complete role inventory checked; per-row feature reconstruction remains separate work.'])
    output = base/'revision_v3/remote_migration/public_road_inventory_20261029.json'
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
