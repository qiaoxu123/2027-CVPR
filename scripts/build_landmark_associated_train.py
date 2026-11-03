"""Materialize a versioned TRAIN-only landmark-associated subset, preserving rows."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from scripts.analyze_instruction_landmark_association import iter_train

KEEP = frozenset({'direct_geometry', 'indirect_geometry', 'direct_identity'})
REMOVE = frozenset({'no_confirmed_anchor', 'uncertain'})


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def task_key(row):
    if len(row['descriptions']) != 1:
        raise ValueError('Expected one official instruction per TRAIN row')
    return f"{row['area']}_block_{row['block']}", row['descriptions'][0]


def decision(record):
    category = record['category']
    if category not in KEEP | REMOVE:
        raise ValueError('Unknown classification: '+str(category))
    return category in KEEP


def classification_index(records):
    lookup = {}
    for record in records:
        decision(record)
        key = record['map_name'], record['instruction']
        old = lookup.get(key)
        if old is not None and (old['category'], old['target_category']) != (record['category'], record['target_category']):
            raise ValueError('Conflicting classification for duplicate map+instruction')
        lookup[key] = record
    return lookup


class JsonArray:
    def __init__(self, path):
        self.stream = path.open('w')
        self.stream.write('[\n')
        self.n = 0

    def append(self, row):
        if self.n:
            self.stream.write(',\n')
        self.stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')))
        self.n += 1

    def close(self):
        self.stream.write('\n]\n')
        self.stream.close()


def build(source, processed, records_path, summary_path, map_manifest, output):
    if output.exists():
        raise FileExistsError('Preserve existing dataset; choose a new output directory')
    if source.name != 'citynav_train_seen.json' or processed.name != 'citynav_train_seen.json':
        raise ValueError('Only explicit TRAIN source files accepted')
    summary = json.loads(summary_path.read_text())
    if sha(source) != summary['provenance']['source_sha256']:
        raise ValueError('Official TRAIN source differs from classified source')
    records = json.loads(records_path.read_text())
    if len(records) != summary['n'] or [r['episode'] for r in records] != list(range(len(records))):
        raise ValueError('Classification order/count mismatch')
    lookup = classification_index(records)
    original_indices = defaultdict(list)
    for record in records:
        original_indices[(record['map_name'], record['instruction'])].append(record['episode'])
    maps = json.loads(map_manifest.read_text())['map_assignment']
    if set(maps.values()) != {'fit', 'dev', 'confirm'}:
        raise ValueError('Existing grouped TRAIN split required')
    output.mkdir(parents=True)
    counts = defaultdict(Counter)
    by_category = defaultdict(Counter)
    original = JsonArray(output/'official_train_associated.json')
    kept = (output/'retained_records.jsonl').open('w')
    removed = (output/'removed_records.jsonl').open('w')
    source_count = 0
    for i, row in iter_train(source):
        r = records[i]
        if task_key(row) != (r['map_name'], r['instruction']):
            raise ValueError('Original index/text mismatch: '+str(i))
        keep = decision(r)
        info = dict(source_index=i, subset_index=original.n if keep else None,
            map_name=r['map_name'], instruction=r['instruction'], object_ids=row['object_ids'],
            target_category=r['target_category'], relation_class=r['category'],
            reason=r.get('reason'), existing_train_partition=maps[r['map_name']])
        (kept if keep else removed).write(json.dumps(info, ensure_ascii=False)+'\n')
        counts['official'][r['category']] += 1
        by_category[r['target_category']]['retained' if keep else 'removed'] += 1
        if keep:
            original.append(row)
        source_count += 1
    original.close(); kept.close(); removed.close()
    if source_count != len(records):
        raise ValueError('Incomplete original classification coverage')

    writers = {part: JsonArray(output/('processed_'+part+'.json')) for part in ('all', 'fit', 'dev', 'confirm')}
    processed_index = (output/'processed_index.jsonl').open('w')
    groups = defaultdict(set)
    total_processed = 0
    for i, row in iter_train(processed):
        key = task_key(row)
        if key not in lookup:
            raise ValueError('Processed row has no exact official text match: '+str(i))
        r = lookup[key]
        keep = decision(r)
        part = maps[key[0]]
        info = dict(source_index=i, matching_official_indices=original_indices[key],
            map_name=key[0], object_ids=row['object_ids'], relation_class=r['category'],
            retained=keep, existing_train_partition=part,
            all_subset_index=writers['all'].n if keep else None,
            partition_subset_index=writers[part].n if keep else None)
        processed_index.write(json.dumps(info)+'\n')
        counts['processed'][r['category']] += 1
        if keep:
            for oid in row['object_ids']:
                groups[part].add((key[0], str(oid)))
            writers['all'].append(row)
            writers[part].append(row)
        total_processed += 1
    processed_index.close()
    for writer in writers.values():
        writer.close()
    for a, b in [('fit','dev'), ('fit','confirm'), ('dev','confirm')]:
        if groups[a] & groups[b]:
            raise ValueError('Physical instance group crosses existing partitions')
    manifest = dict(version='landmark-associated-train-v1', complete=True,
        keep_categories=sorted(KEEP), remove_categories=sorted(REMOVE),
        classification='Frozen full-corpus automatic dependency classification; no sample-review overrides',
        official_total=source_count, official_retained=original.n, official_removed=source_count-original.n,
        processed_total=total_processed, processed_retained=writers['all'].n,
        processed_removed=total_processed-writers['all'].n,
        counts={k:dict(v) for k,v in counts.items()}, by_target_category=dict(by_category),
        partition_sizes={k:v.n for k,v in writers.items() if k!='all'},
        partition_object_id_groups={k:len(v) for k,v in groups.items()},
        recommended_fit_annotations=str((output/'processed_fit.json').resolve()),
        warnings=['Automatic relation labels are not human verified.',
            'Selected TRAIN distribution excludes ambiguous instructions; formal evaluation must remain unfiltered.',
            'Old TRAIN dev/confirm partitions retain their prior exposure history; not fresh independent data.',
            'Source indices change after filtering: use processed_index.jsonl; never reuse positional caches without mapping.',
            'Target coordinates, trajectory and IDs are preserved as annotation fields, not new inference inputs.'],
        source_sha256={str(p.resolve()):sha(p) for p in (source, processed, records_path, summary_path, map_manifest, Path(__file__))},
        files={p.name:dict(sha256=sha(p), bytes=p.stat().st_size) for p in sorted(output.iterdir()) if p.is_file()})
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source','processed','records','summary','map-manifest','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.source, args.processed, args.records, args.summary, args.map_manifest, args.output)
    print(json.dumps({k:manifest[k] for k in ('official_total','official_retained','official_removed',
        'processed_total','processed_retained','processed_removed','partition_sizes','counts','by_target_category')}, indent=2))


if __name__ == '__main__':
    main()
