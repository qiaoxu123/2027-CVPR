"""Verify complete on-policy evidence against immutable scored rollout traces."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from graphnav.checkpointing import atomic_json,file_sha256


def validate(collection):
    root=Path(collection)
    manifest=json.loads((root/'collection.json').read_text());episodes=json.loads((root/'episodes.json').read_text());traces=[json.loads(s) for s in (root/'predictions.jsonl').read_text().splitlines()]
    if not manifest.get('complete') or manifest['partition']!='train' or not len(episodes)==len(traces)==len(manifest['episodes']):raise ValueError('Incomplete Train collection')
    total=0;near=0
    for episode,label,trace in zip(episodes,manifest['episodes'],traces):
        if not episode['index']==label['index']==trace['index'] or not episode['episode_id']==label['episode_id']==trace['episode_id']:raise ValueError('Episode identity mismatch')
        path=root/label['file']
        if file_sha256(path)!=label['sha256']:raise ValueError('Evidence checksum mismatch')
        rows=torch.load(path,map_location='cpu',weights_only=True)
        if len(rows)!=len(trace['step_diagnostics']) or len(rows)!=label['states']:raise ValueError('Decision count mismatch')
        for row,diagnostic,pose in zip(rows,trace['step_diagnostics'],trace['poses']):
            if row['step']!=diagnostic['observation_step'] or not np.allclose(row['pose'],pose,atol=1e-4):raise ValueError('Decision pose/step mismatch')
            if any('target' in key or 'label' in key for key in row):raise ValueError('Private labels leaked into evidence')
            frame=row['frame']
            for key in ['features','geometry','query_world','kernel']:
                if not torch.isfinite(frame[key].float()).all():raise ValueError('Nonfinite evidence')
            total+=1;near+=int(np.linalg.norm(np.asarray(row['pose'])[:2]-np.asarray(label['target_world'])[:2])<=20)
    result=dict(passed=True,episodes=len(episodes),states=total,near_goal_states=near,collection_sha=file_sha256(root/'collection.json'),trace_sha=file_sha256(root/'predictions.jsonl'),protocol='Pose/step/episode alignment, evidence hashes, no top-level private label fields, finite public features. Group separation separately enforced before collection.')
    atomic_json(result,root/'validation.json')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--collection',required=True);a=p.parse_args()
    print(json.dumps(validate(a.collection)))


if __name__=='__main__':main()
