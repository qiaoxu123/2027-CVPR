"""Reproducible map-balanced TRAIN pilot selection; GT is not used to select rows."""
import argparse,collections,json,random
from pathlib import Path
from graphnav.checkpointing import file_sha256,atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=1000);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve immutable pilot input')
    groups=collections.defaultdict(list)
    for line in a.source.open():
        row=json.loads(line)
        if row.get('dataset_version')!='revised':raise ValueError('Revised TRAIN required')
        groups[row['map_name']].append(row)
    if not 1<=a.limit<=sum(map(len,groups.values())):raise ValueError('Invalid pilot limit')
    rng=random.Random(a.seed)
    for k in sorted(groups):rng.shuffle(groups[k])
    selected=[]
    while len(selected)<a.limit:
        for k in sorted(groups):
            if groups[k] and len(selected)<a.limit:selected.append(groups[k].pop())
    selected.sort(key=lambda x:x['map_name']);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(''.join(json.dumps(d)+'\n' for d in selected))
    atomic_json(dict(source_sha256=file_sha256(a.source),output_sha256=file_sha256(a.output),seed=a.seed,n=len(selected),maps=dict(collections.Counter(d['map_name'] for d in selected)),selection='map-balanced round-robin, random within map; no target-conditioned selection'),a.output.with_suffix('.manifest.json'))
if __name__=='__main__':main()
