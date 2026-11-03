"""Encode public reference phrases only; original image cache remains unchanged."""
import argparse
import json
from pathlib import Path
import torch
from graphnav.appearance_matching import ObservedSigLIPMatchingEncoder
from graphnav.checkpointing import atomic_save,atomic_json,file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('roles','public-frames','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--backbone',required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('do not overwrite role vectors')
    roles=json.loads(a.roles.read_text());frames={f['task_key']:f for f in map(json.loads,a.public_frames.read_text().splitlines())}
    native=ObservedSigLIPMatchingEncoder(a.backbone)
    keys=list(roles);vectors={};public=[]
    for start in range(0,len(keys),32):
        ids=keys[start:start+32];encoded=native.texts([roles[k]['reference_text'] for k in ids]).cpu().half()
        for i,key in enumerate(ids):
            r=roles[key];vectors[key]=dict(reference_vector=encoded[i].clone(),target_phrase=r['target_text'])
            public.append(dict(map_name=frames[key]['map_name'],instruction=frames[key]['instruction'],
                               target_text=r['target_text'],context_text=r['reference_text'],version='local-roles-v2'))
    atomic_save(vectors,a.output)
    atomic_json(public,a.output.with_suffix('.public.json'))
    atomic_json(dict(roles_sha256=file_sha256(a.roles),public_frames_sha256=file_sha256(a.public_frames),
        vectors_sha256=file_sha256(a.output),tasks=len(roles),
        with_reference_phrases=sum(bool(r.get('reference_phrases')) for r in roles.values()),
        indirect_paths=sum(any(p['hops']>=2 for p in r['parse'].get('paths',[])) for r in roles.values()),
        note='Local dependency proposals and public map types, not verified semantic gold.'),a.output.with_suffix('.provenance.json'))


if __name__=='__main__':main()
