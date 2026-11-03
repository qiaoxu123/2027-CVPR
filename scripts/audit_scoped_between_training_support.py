"""Seal public map-only counterfactuals before scoring private TRAIN targets."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from graphnav.checkpointing import atomic_json,atomic_save,file_sha256
from graphnav.scoped_between_geometry import between_distance
from graphnav.visual_chain_position import WorldLikelihoodPositionHead
from scripts.train_local_chain_position import INPUTS,collate,batch_inputs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','observations','roles','filter','corpus','catalog','checkpoint','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(4)
    meta=json.loads((a.cache/'manifest.json').read_text());assert meta['complete'] and meta['split']=='train_seen'
    assert meta['public_catalog_sha256']==file_sha256(a.catalog)
    allowed=set(json.loads(a.filter.read_text())['frame_keys'])
    frames={x['frame_key']:x for x in map(json.loads,(a.observations/'public_frames.jsonl').read_text().splitlines())}
    corpus={(x['map_name'],x['instruction']):x for x in json.loads(a.corpus.read_text())}
    objects=json.loads(a.catalog.read_text());roles=torch.load(a.roles,weights_only=True)
    public=[]
    for c in meta['chunks']:
        path=a.cache/c['file'];assert file_sha256(path)==c['sha256']
        for r in torch.load(path,weights_only=True)['records']:
            if r['frame_key'] not in allowed:continue
            assert r['subset']=='fit' and not {'target_world','target_id','physical_group'}&r.keys()
            frame=frames[r['frame_key']];proposal=corpus[(frame['map_name'],frame['instruction'])]
            costs=[]
            for g in proposal['groups']:
                if g['kind']!='two-public-anchors':continue
                cost,known=between_distance(r['candidate_world'].numpy(),g['endpoints'],objects[frame['map_name']])
                if known:costs.append(cost)
            r['context_text']=roles[r['task_key']]['reference_vector'].clone()
            r['between_cost']=torch.from_numpy(np.mean(costs,axis=0) if costs else np.zeros(928,np.float32))
            r['between_known']=bool(costs)
            public.append(r)
    model=WorldLikelihoodPositionHead().cuda().eval().requires_grad_(False)
    model.load_state_dict(torch.load(a.checkpoint,weights_only=True,map_location='cpu')['model'])
    predictions={}
    # This inference stage has not opened labels_private.jsonl.
    with torch.inference_mode():
        for start in range(0,len(public),8):
            rows=public[start:start+8];x={k:torch.stack([r[k] for r in rows]).cuda() for k in INPUTS}
            v3=model(**x)[:,:784].cpu();prior=x['prior_logits'].cpu()
            for i,r in enumerate(rows):
                energy=-r['between_cost'][:784]/20.
                predictions[r['frame_key']]=dict(public_between_known=r['between_known'],
                    chosen={m:int(z.argmax()) for m,z in dict(A0=prior[i],V3=v3[i],A0Between=prior[i]+energy,V3Between=v3[i]+energy).items()},
                    between_cost=r['between_cost'],local_valid=r['local_valid'],candidate_world=r['candidate_world'])
    a.output.mkdir(parents=True);atomic_save(predictions,a.output/'predictions_public.pt')
    atomic_json(dict(predictions_sha256=file_sha256(a.output/'predictions_public.pt'),private_labels_read=False,
        sigma_m=20.,warning='One fixed coarse map-only proxy; not a deployable visual method or oracle optimum.'),a.output/'prediction_seal.json')
    # Target coordinates below are TRAIN posthoc diagnostics only.
    truth={x['frame_key']:x for x in map(json.loads,(a.observations/'labels_private.jsonl').read_text().splitlines()) if x['status']=='captured'}
    rows=[]
    for key,pred in predictions.items():
        label=truth[key];target=torch.tensor(label['target_world'][:2]);world=pred['candidate_world']
        distances=(world-target).norm(dim=-1);valid=pred['local_valid'];close=distances[784:][valid]
        good=distances[:784]<=20.;cost=pred['between_cost'][:784]
        rows.append(dict(frame_key=key,task_key=label['task_key'],category=label['category'],
            local20_support=bool(len(close) and close.min()<=20),local5_support=bool(len(close) and close.min()<=5),
            between_known=pred['public_between_known'],global_hull_fraction=float((cost<=.01).float().mean()),
            correct20_in_hull=bool(((cost<=.01)&good).any()),
            methods={m:dict(ne_m=float(distances[c]),success=bool(distances[c]<=20)) for m,c in pred['chosen'].items()}))
    summary=dict(frames=len(rows),targets=len({r['task_key'] for r in rows}),
        local20_support=sum(r['local20_support'] for r in rows),local5_support=sum(r['local5_support'] for r in rows),
        correct20_in_hull=sum(r['correct20_in_hull'] for r in rows),median_global_hull_fraction=float(np.median([r['global_hull_fraction'] for r in rows])),methods={})
    for name in ('A0','V3','A0Between','V3Between'):
        base='A0' if name.startswith('A0') else 'V3'
        summary['methods'][name]=dict(correct=sum(r['methods'][name]['success'] for r in rows),
            r1=float(np.mean([r['methods'][name]['success'] for r in rows])),ne_m=float(np.mean([r['methods'][name]['ne_m'] for r in rows])),
            corrected=sum(r['methods'][name]['success'] and not r['methods'][base]['success'] for r in rows),
            harmed=sum(not r['methods'][name]['success'] and r['methods'][base]['success'] for r in rows))
    summary['public_inference_sealed_before_private_scoring']=True
    summary['sha256']={k:file_sha256(v) for k,v in dict(checkpoint=a.checkpoint,roles=a.roles,filter=a.filter,corpus=a.corpus,producer=Path(__file__),geometry=Path('graphnav/scoped_between_geometry.py')).items()}
    atomic_json(rows,a.output/'scored_private.json');atomic_json(summary,a.output/'summary.json');print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
