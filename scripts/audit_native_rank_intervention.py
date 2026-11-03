"""TRAIN controlled native-rank experiment plus explicitly weak-box oracle."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from graphnav.checkpointing import atomic_json,atomic_save,file_sha256
from graphnav.visual_rank_permutation import native_rank_permutation,weak_box_oracle_permutation,replace_visual_world_field


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('selection','forward','annotations','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--weak-box-oracle',action='store_true');a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    source=json.loads((a.forward/'prediction_seal.json').read_text());assert source['predictions_sha256']==file_sha256(a.forward/'predictions_public.pt')
    queries=torch.load(a.selection/'inputs_public.pt',weights_only=True);pred=torch.load(a.forward/'predictions_public.pt',weights_only=True)
    normal={}
    for key,q in queries.items():
        normal[key]={}
        for mode in ('Native144','Native576'):
            x=pred[key][mode];bounds=q['world_bounds'];uv=(x['world']-bounds[:2])/(bounds[2:]-bounds[:2])
            scores=native_rank_permutation(x['visual_score'],x['cosine'],x['valid'])
            assert torch.equal(scores[x['valid']].sort().values,x['visual_score'][x['valid']].sort().values)
            normal[key][mode]=replace_visual_world_field(x['logprob'],uv,x['visual_score'],scores,x['valid'])
    a.output.mkdir(parents=True);atomic_save(normal,a.output/'native_rank_predictions_public.pt')
    atomic_json(dict(normal_predictions_sha256=file_sha256(a.output/'native_rank_predictions_public.pt'),private_weak_boxes_read=False,private_goals_read=False),a.output/'normal_prediction_seal.json')
    annotations={x['frame_key']:x for x in json.loads(a.annotations.read_text())};truth={}
    for obs in {q['observations'] for q in queries.values()}:
        for x in map(json.loads,(Path(obs)/'labels_private.jsonl').read_text().splitlines()):
            if x.get('frame_key') in queries:truth[x['frame_key']]=x
    rows=[]
    for key,q in queries.items():
        target=torch.tensor(truth[key]['target_world'][:2]);world=q['global_world'];results={}
        for mode in ('Native144','Native576'):
            x=pred[key][mode];variants=dict(Original=x['logprob'],NativeRank=normal[key][mode])
            if a.weak_box_oracle:
                box=annotations[key]['boxes_px'][0];xy=x['pixels'];positive=(xy[:,0]>=box[0])&(xy[:,0]<box[2])&(xy[:,1]>=box[1])&(xy[:,1]<box[3])
                new=weak_box_oracle_permutation(x['visual_score'],positive,x['valid'],diagnostic=True)
                uv=(x['world']-q['world_bounds'][:2])/(q['world_bounds'][2:]-q['world_bounds'][:2])
                variants['WeakBoxOracle']=replace_visual_world_field(x['logprob'],uv,x['visual_score'],new,x['valid'])
            results[mode]={}
            for name,field in variants.items():
                goal=world[field.argmax()];ne=float((goal-target).norm());results[mode][name]=dict(ne_m=ne,success=ne<=20,goal_world=goal.tolist())
        rows.append(dict(frame_key=key,results=results))
    summary={}
    for mode in ('Native144','Native576'):
        summary[mode]={}
        for name in rows[0]['results'][mode]:
            values=[r['results'][mode][name] for r in rows];base=[r['results'][mode]['Original'] for r in rows]
            summary[mode][name]=dict(correct=sum(x['success'] for x in values),ne_m=float(np.mean([x['ne_m'] for x in values])),
                corrected=sum(x['success'] and not b['success'] for x,b in zip(values,base)),harmed=sum(not x['success'] and b['success'] for x,b in zip(values,base)))
    atomic_json(dict(summary=summary,rows=rows,normal_branch_reads_gt=False,weak_oracle_enabled=a.weak_box_oracle,
        warning='Annotation-selected TRAIN27 appearance-proxy diagnosis; weak-box oracle is not true-instance gold or deployable performance.',producer_sha256=file_sha256(__file__)),a.output/'results.json')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
