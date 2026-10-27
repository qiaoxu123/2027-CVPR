"""Audit all CityNav train queries before detector extraction (CPU only)."""
from collections import Counter, defaultdict
import time
from .paths import bootstrap, DATA, ROOT, BERT_REVISION, save, sha, verify_frozen_assets
from .data_protocol import split_target_groups, resolve_public_references
from .mentions import landmark_mention_spans


def main():
    bootstrap()
    from sbfnav.dataset import CityNavDataset, CityReferCatalog
    from transformers import AutoTokenizer
    start = time.perf_counter()
    assets = verify_frozen_assets()
    save(assets, ROOT/'reports/a2_execution_20261027/frozen_sources.json')
    ds = CityNavDataset(DATA, 'train_seen', version='processed_citynav')
    catalog = CityReferCatalog(DATA/'cityrefer')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', revision=BERT_REVISION, local_files_only=True, use_fast=True)
    splits = split_target_groups(ds, seed=0)
    groups, counters, entries = defaultdict(set), defaultdict(Counter), []
    named = {name: [o for o in objects.values() if o.get('name')] for name, objects in catalog.objects.items()}
    for i, (row, split) in enumerate(zip(ds, splits)):
        names = catalog.referenced_landmark_names(row)
        refs = resolve_public_references(row.map_name, names, named[row.map_name])
        tokens = tokenizer(row.instruction, truncation=True, max_length=96, return_offsets_mapping=True)
        _, valid = landmark_mention_spans(row.instruction, tokens['offset_mapping'], refs)
        count = counters[split]
        count.update(episodes=1, reference_queries=len(names), reference_records=len(refs),
                     linked_references=sum(bool(r['linked']) for r in refs),
                     token_aligned_references=int(valid.sum()), episodes_with_link=int(valid.any()),
                     truncated_instructions=int(len(tokenizer(row.instruction)['input_ids']) > 96))
        count['target_type:'+catalog.target_object(row)['object_type']] += 1
        groups[split].add((row.map_name, row.object_id))
        entries.append(dict(index=i, map_name=row.map_name, object_id=row.object_id,
                            description_id=row.description_id, split=split,
                            reference_queries=len(names), linked_references=int(valid.sum())))
        if (i+1) % 2000 == 0:
            print(f'audited {i+1}/{len(ds)}', flush=True)
    assert not (groups['train'] & groups['dev'] or groups['train'] & groups['calibration'] or groups['dev'] & groups['calibration'])
    save(dict(seed=0, annotation_path=str(ds.annotation_path), annotation_sha256=sha(ds.annotation_path),
              split_unit='map + target object id', entries=entries), ROOT/'runs/a2_waldo_landmark_20261027/source_manifest.json')
    report = dict(status='source_audit_complete_detector_funnel_pending', episodes=len(ds),
                  target_instances=sum(map(len, groups.values())),
                  split_counts={s: dict(c, target_instances=len(groups[s])) for s,c in counters.items()},
                  group_disjoint=True, source_manifest_sha256=sha(ROOT/'runs/a2_waldo_landmark_20261027/source_manifest.json'),
                  seconds=time.perf_counter()-start,
                  limitations=['Detector coverage and reliable positive/negative associations not yet measured.',
                              'Public name homonyms and unresolved aliases remain masked; no fuzzy identity assignment.'])
    save(report, ROOT/'reports/a2_execution_20261027/source_audit.json')
    print(report, flush=True)


if __name__ == '__main__':
    main()
