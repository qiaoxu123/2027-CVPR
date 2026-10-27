"""CPU-only all-row audit before training a user-authorized collected snapshot."""
import argparse
from collections import Counter
import json
from .paths import ROOT,save,sha
from .collected_snapshot import BASE,cache_source,snapshot_partition,verify_cache_origin
from .training_protocol import validate_complete_cache
from .train import CachedViews
from .prepare import view_counts


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cache',type=__import__('pathlib').Path,required=True)
    args=parser.parse_args()
    verify_cache_origin(args.cache);source=cache_source(args.cache)
    manifest=validate_complete_cache(args.cache,source)
    counters={};groups={};maps={}
    for partition in ('train','dev','calibration'):
        dataset=CachedViews(args.cache,manifest,source,partition);counter=Counter();targets=set();map_counts=Counter()
        for i in range(len(dataset)):
            row=dataset[i];map_name,oid=row['target_instance'];targets.add((map_name,oid));map_counts[map_name]+=1
            for j,detail in enumerate(row['private']['association']):
                if row['labels']['rank_label_mask'][j]:
                    if detail['object_id'] is None or snapshot_partition(map_name,detail['object_id'])!=partition:
                        raise ValueError('Cross-partition candidate supervision leaked')
            counter.update(view_counts(row))
            if (i+1)%2000==0:print(f'Audited {partition} {i+1}/{len(dataset)}',flush=True)
        if counter['informative_views']<=0:raise ValueError(f'No informative contrasts in {partition}')
        if counter['offset_positive_views']<=0:raise ValueError(f'No reachable offset positives in {partition}')
        groups[partition]=targets;counters[partition]=dict(counter);maps[partition]=dict(map_counts)
    if any(groups[a]&groups[b] for a,b in [('train','dev'),('train','calibration'),('dev','calibration')]):
        raise ValueError('Targets overlap across partitions')
    report=dict(status='all_snapshot_rows_and_partition_labels_verified',counters=counters,views_by_map=maps,
                target_counts={p:len(g) for p,g in groups.items()},target_intersection=0,
                cross_partition_supervision=0,signature_sha256=sha(args.cache/'signature.json'),
                manifest_sha256=sha(args.cache/'manifest.json'),calibration_on_policy=False)
    save(report,BASE/'snapshot_audit.json');save(report,ROOT/'reports/a2_execution_20261027/SNAPSHOT_AUDIT.json')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
