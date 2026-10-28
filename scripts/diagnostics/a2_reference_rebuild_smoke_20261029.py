"""Actual frozen CPU BERT + checked TRAIN/DEV sources; no cache writes."""
import json
import sys
from pathlib import Path

import torch

from experiments.a2_waldo_landmark_belief.paths import DATA, SOURCE, BERT_REVISION, sha
from experiments.a2_waldo_landmark_belief.relation_v3_data import RelationV3Views, read_training_source
from experiments.a2_waldo_landmark_belief.public_road_references import resolve_road_references
from experiments.a2_waldo_landmark_belief.reference_rebuild import rebuild_reference_pack
from experiments.a2_waldo_landmark_belief.reference_training import build_reference_training_row


def main():
    torch.set_num_threads(2)
    # Append only: prepending GraphNav's scripts directory shadows this project.
    sys.path.append(str(SOURCE))
    from graphnav.token_language import FrozenBERTTokens
    bert = FrozenBERTTokens(device='cpu')
    if bert.revision != BERT_REVISION:
        raise ValueError('Frozen BERT revision differs')
    base = Path('runs/a2_waldo_landmark_20261027')
    source = json.loads((base/'source_manifest.json').read_text())
    if sha(source['annotation_path']) != source['annotation_sha256']:
        raise ValueError('Source instruction annotation changed')
    records = json.loads(Path(source['annotation_path']).read_text())
    objects = json.loads((DATA/'cityrefer/objects.json').read_text())
    processed = json.loads((DATA/'cityrefer/processed_descriptions.json').read_text())
    populations = {'teacher_snapshot': {e['index']:e for e in source['entries']}}
    for version in ('revision_v1','revision_v2'):
        pop = json.loads((base/version/'confirmation_population.json').read_text())
        populations[version+'_A0_closedloop'] = {e['index']:e for e in pop['entries']}
    results = []
    modified = {'mention_mask','landmark_name_tokens','name_valid','link_valid','pair_geometry','pair_valid'}
    for role in ('train','dev'):
        dataset = RelationV3Views(role)
        samples, seen = [],set()
        for index,entry in enumerate(dataset.entries):
            if entry['source_kind'] not in seen:
                seen.add(entry['source_kind']); samples.append(index)
        if role == 'dev':
            audit = json.loads((base/'revision_v3/remote_migration/epoch04_public_topology_audit.json').read_text())
            samples.append(next(c['dev_index'] for c in audit['cases'] if c['hypothetical_public_mention_unblocked']))
        for index in dict.fromkeys(samples):
            entry = dataset.entries[index]
            raw = read_training_source(entry,role,dataset.allowed,dataset.contexts[entry['source_kind']]['signature'])
            record = records[entry['episode']]
            identity = populations[entry['source_kind']][entry['episode']]
            expected = (entry['map_name'],entry['object_id'],identity['description_id'])
            if (f"{record['area']}_block_{record['block']}",int(record['object_ids'][0]),int(record['ann_ids'][0])) != expected:
                raise ValueError('Source instruction identity mismatch')
            instruction = str(record['descriptions'][0])
            if 'instruction' in raw and raw['instruction'] != instruction:
                raise ValueError('Cached instruction differs from bound annotation')
            queries = processed[entry['map_name']][str(entry['object_id'])][identity['description_id']].get('landmarks',[])
            named = [o for o in objects[entry['map_name']].values() if o.get('name')]
            refs = resolve_road_references(entry['map_name'],queries,named)
            offsets = bert.tokenizer(instruction,truncation=True,max_length=96,return_offsets_mapping=True)['offset_mapping']
            public,meta = rebuild_reference_pack(raw['public'],raw['meta'],instruction=instruction,
                offsets=offsets,references=refs,encode_names=bert.encode_tokens)
            for key,value in raw['public']['network'].items():
                if key not in modified and not torch.equal(value,public['network'][key]):
                    raise ValueError('Immutable public network input changed')
            for key,value in raw['public']['spatial'].items():
                if not torch.equal(value,public['spatial'][key]):
                    raise ValueError('Immutable public spatial input changed')
            if sha(Path(entry['directory'])/entry['filename']) != entry['sha256']:
                raise ValueError('Parent cache file changed')
            if not torch.isfinite(public['network']['pair_geometry']).all():
                raise ValueError('Nonfinite rebuilt geometry')
            old_refs = {(r['map_name'],str(r['object_id'])):i for i,r in enumerate(raw['meta']['references'])}
            parity_count = 0
            for j,ref in enumerate(meta['references']):
                key = (ref['map_name'],str(ref['object_id']))
                if 'region' not in ref and key in old_refs:
                    if not torch.equal(raw['public']['network']['pair_geometry'][:,old_refs[key]], public['network']['pair_geometry'][:,j]):
                        raise ValueError('Legacy singleton pair features changed')
                    parity_count += 1
            supervised = build_reference_training_row(dict(public=public,meta=meta,labels=raw['labels']),
                target_kind=dataset.target_kinds[(entry['map_name'],entry['object_id'])],
                real=entry['source_kind']!='teacher_snapshot')
            if supervised['labels']['angle'].shape[0] != len(public['network']['link_valid']):
                raise ValueError('Rebuilt TRAIN label/reference axes differ')
            results.append(dict(role=role,source_kind=entry['source_kind'],filename=entry['filename'],
                old_references=raw['meta']['valid_references'],new_references=meta['valid_references'],
                grouped_references=sum('region' in ref for ref in meta['references']),
                valid_pairs=int(public['network']['pair_valid'].sum()),
                name_feature_shape=list(public['network']['landmark_name_tokens'].shape),
                singleton_geometry_parity_count=parity_count,
                spatially_usable_candidates=int(supervised['labels']['localization_utility'].sum()),
                parent_sha256=entry['sha256'],parent_unchanged=True))
    if torch.cuda.is_initialized():
        raise ValueError('CPU-only smoke test unexpectedly opened CUDA')
    report = dict(status='actual_cpu_reference_rebuild_smoke_passed',bert_revision=bert.revision,
                  examples=results,no_gpu_used=True,original_cache_unchanged=True,
                  full_cache_rebuilt=False,training_changed=False,benchmark_used=False)
    output = base/'revision_v3/remote_migration/reference_rebuild_smoke_20261029.json'
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
