"""Merge completed Train on-policy evidence with teacher prefixes, preserving grouping."""
import argparse,json,hashlib
from pathlib import Path
import torch
from graphnav.checkpointing import atomic_json,atomic_save,file_sha256
from scripts.train_object_query_belief import load
from scripts.validate_native_collection import validate


def main():
    p=argparse.ArgumentParser();p.add_argument('--teacher',required=True);p.add_argument('--collection',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise ValueError('Refusing overwrite')
    validate(a.collection)
    meta,rows=load(a.teacher);directory=Path(a.collection);m=json.loads((directory/'collection.json').read_text())
    if meta['partition']!='train' or m['partition']!='train' or not m['complete'] or meta['base_sha']!=m['base_sha']:raise ValueError('Completed matching Train sources required')
    additions=[]
    for episode in m['episodes']:
        path=directory/episode['file']
        if file_sha256(path)!=episode['sha256']:raise ValueError('Evidence checksum mismatch')
        group=hashlib.sha256(json.dumps((episode['map_name'],episode['instruction'])).encode()).hexdigest()
        for evidence in torch.load(path,map_location='cpu',weights_only=True):
            frame=evidence['frame'];row={k:v[0].clone() for k,v in frame.items() if k!='proposal_ids'}
            row.update(base_logits=evidence['base_logits'][0].float(),grid_world=evidence['grid_world'].float(),pose=torch.as_tensor(evidence['pose']).float(),labels=dict(target_world=torch.tensor(episode['target_world'],dtype=torch.float32)),episode=episode['index'],step=evidence['step'],group=group,instruction=episode['instruction'],map_name=episode['map_name'],origin='onpolicy')
            # Store one common kernel convention; training recomputes alternate sigma.
            row['kernel']=torch.exp(-torch.cdist(row['grid_world'],row['query_world']).square()/(2*meta['kernel_sigma_m']**2))*row['observed'][:,None]*row['valid'][None]
            row['features']=row['features'].half();row['text']=row['text'].half();row['kernel']=row['kernel'].half()
            additions.append(row)
    # Initial observations are identical; avoid giving them twice the weight.
    initial={r['episode'] for r in additions if r['step']==0}
    rows=[r for r in rows if not(r['step']==0 and r['episode'] in initial)]+additions
    out.mkdir(parents=True);chunks=[]
    for start in range(0,len(rows),32):
        name=f'rows_{len(chunks):04d}.pt';part=rows[start:start+32];atomic_save(part,out/name);chunks.append(dict(file=name,rows=len(part),sha256=file_sha256(out/name)))
    atomic_json(dict(complete=True,partition='train',base_sha=meta['base_sha'],kernel_sigma_m=meta['kernel_sigma_m'],chunks=chunks,states=len(rows),onpolicy_states=len(additions),teacher_manifest_sha=file_sha256(Path(a.teacher)/'manifest.json'),collection_sha=file_sha256(directory/'collection.json'),rows=[dict(episode=r['episode'],step=r['step'],group=r['group'],origin=r.get('origin','teacher')) for r in rows]),out/'manifest.json')
    print(json.dumps(dict(states=len(rows),onpolicy_states=len(additions))))


if __name__=='__main__':main()
