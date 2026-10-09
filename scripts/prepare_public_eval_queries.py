"""Prepare only supplied language/map inputs, keeping evaluation labels separate."""
import argparse
import json
from pathlib import Path
import sys
from graphnav.a3_runtime import public_static_record
from graphnav.closedloop import PublicEpisode,PublicCatalog
from graphnav.checkpointing import file_sha256,provenance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('data-root','sbf-source','config','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--split',choices=['val_seen'],default='val_seen');args=p.parse_args();out=Path(args.output)
    if out.exists():raise ValueError('refusing public-evaluation query overwrite')
    out.mkdir(parents=True);sys.path.insert(0,args.sbf_source)
    from sbfnav.dataset import CityNavDataset,CityReferCatalog
    from sbfnav.planning_state import PlanningStateBuilder
    from sbfnav.config import load_config
    config=load_config(args.config)
    dataset=CityNavDataset(args.data_root,args.split,version=config['data']['version']);catalog=CityReferCatalog(Path(args.data_root)/'cityrefer')
    builder=PlanningStateBuilder(args.data_root,PublicCatalog(catalog))
    with (out/'public_queries.jsonl').open('w') as inputs,(out/'labels.jsonl').open('w') as labels:
        for index in range(len(dataset)):
            row=dataset[index]
            episode=PublicEpisode(row.map_name,row.instruction,row.start_pose.copy(),
                tuple(catalog.referenced_landmarks(row)),tuple(catalog.referenced_landmark_names(row)))
            query=public_static_record(episode,builder.transform(row.map_name),catalog)
            inputs.write(json.dumps(query)+'\n')
            labels.write(json.dumps(dict(query_index=index,source_episode_index=row.index,map_name=row.map_name,
                target_world=row.target_position.tolist()))+'\n')
    report=dict(split=args.split,episodes=len(dataset),provenance=provenance(),config_sha256=file_sha256(args.config),version=config['data']['version'],
        source_sha256=file_sha256(dataset.annotation_path),public_queries_sha256=file_sha256(out/'public_queries.jsonl'),
        labels_sha256=file_sha256(out/'labels.jsonl'),note='Public dummy target[0,0] packing only. Private destination labels stay separate; no deduplication/drop of benchmark episodes.')
    (out/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
