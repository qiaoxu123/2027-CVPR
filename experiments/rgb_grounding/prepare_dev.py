"""Select internal Train Dev families; no CityNav validation labels are read."""
import argparse
import hashlib
import json
from pathlib import Path
from experiments.bidirectional_core.data_pilot import balanced
from graphnav.checkpointing import file_sha256


def run(args):
    source=Path(args.source)/'train_seen';root=Path(args.output)/'train_seen'
    if root.exists():raise ValueError('refusing Dev metadata overwrite')
    root.mkdir(parents=True)
    candidates=[]
    with (source/'episodes.jsonl').open() as handle:
        for line in handle:
            r=json.loads(line)
            if r['direction']=='forward' and r['internal_partition']=='internal_dev':candidates.append(r)
    chosen=balanced(candidates,args.families);families={r['family_id'] for r in chosen}
    counts={}
    for source_name,output_name in [('episodes.jsonl','pilot_episodes.jsonl'),('prefixes.jsonl','pilot_prefixes.jsonl'),('matched_e0_e1.jsonl','pilot_matched_e0_e1.jsonl')]:
        n=0
        with (source/source_name).open() as ih,(root/output_name).open('w') as oh:
            for line in ih:
                r=json.loads(line)
                if r['family_id'] in families:oh.write(line);n+=1
        counts[output_name]=n
    pilot=[json.loads(x) for x in (source/'pilot_episodes.jsonl').read_text().splitlines()]
    trained_families={r['family_id'] for r in pilot};trained_maps={r['map_name'] for r in pilot}
    if families & trained_families or {r['map_name'] for r in chosen}&trained_maps:
        raise ValueError('Train/Dev complete-component disjointness failed')
    manifest=dict(families=len(families),maps=sorted({r['map_name'] for r in chosen}),counts=counts,
        train_family_overlap=0,train_map_overlap=0,selection='SHA family and map round robin, only original internal_dev',
        full_partition_manifest_sha256=file_sha256(Path(args.source)/'manifest.json'))
    (Path(args.output)/'selection.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--families',type=int,default=128);run(p.parse_args())


if __name__=='__main__':main()
