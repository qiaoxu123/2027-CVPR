"""Audit named-reference coverage using only TRAIN/internal-Dev public geometry."""
import argparse
from collections import Counter
from pathlib import Path

from graphnav.checkpointing import atomic_json
from graphnav.native_relation_grounding import public_relation_descriptors
from graphnav.semantic_geometry import RELATIONS
from graphnav.stage1_data import Stage1Dataset
from scripts.train_object_query_belief import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('train', 'dev', 'stage1-cache', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise ValueError('Use a new diagnostic output')
    static = Stage1Dataset(args.stage1_cache, verify=False).static
    index = {}
    for i, record in enumerate(static.records):
        index.setdefault((record['map_name'], record['instruction']), []).append(i)
    report = {}
    for name, folder, partition in [('train', args.train, 'train'), ('internal_dev', args.dev, 'dev')]:
        manifest, rows = load(folder)
        if manifest.get('partition') != partition or not manifest.get('complete'):
            raise ValueError('Only complete TRAIN/internal Dev permitted')
        kinds, groups = Counter(), set()
        ambiguous = geometry_mismatch = 0
        for row in rows:
            key = (row['map_name'], row['instruction'])
            ids = index[key]
            graph = static._prepared(ids[0])[0]
            _, types, valid = public_relation_descriptors(row['query_world'].float().numpy(), graph)
            kinds.update(RELATIONS[int(k)] for k in types[valid])
            if key not in groups:
                groups.add(key)
                ambiguous += sum(any(len(a) != 1 for a in rel.anchors) for rel in graph['relations'])
                first, last = static.records[ids[0]], static.records[ids[-1]]
                geometry_mismatch += any(first.get(k) != last.get(k) for k in
                                         ['landmarks', 'world_bounds', 'physical_map_sizes'])
        report[name] = dict(valid_relation_kind_states=dict(kinds), distinct_instruction_keys=len(groups),
                            ambiguous_relation_groups=ambiguous,
                            first_last_public_metadata_mismatches=geometry_mismatch)
    atomic_json(report, Path(args.output))


if __name__ == '__main__':
    main()
