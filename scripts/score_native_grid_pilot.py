"""Post-seal TRAIN AI weak-box proxies and registered-goal diagnostics."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from graphnav.checkpointing import atomic_json,file_sha256
from scripts.ai_weak_box_schema import validate_weak_box_record


def ranking_proxy(pred,box,score_key,delta=0):
    x0,y0,x1,y1=box;box=[max(0,x0-delta),max(0,y0-delta),min(384,x1+delta),min(384,y1+delta)]
    xy=pred['pixels'];valid=pred['valid'];positive=(xy[:,0]>=box[0])&(xy[:,0]<box[2])&(xy[:,1]>=box[1])&(xy[:,1]<box[3])
    target=positive&valid;score=pred[score_key].masked_fill(~valid,float('-inf'))
    order=torch.argsort(score,descending=True,stable=True);order=order[valid[order]]
    hits=target[order].nonzero().flatten();rank=int(hits[0])+1 if len(hits) else None
    return dict(raw_center_support=bool(positive.any()),candidate_present=bool(target.any()),valid_candidates=int(valid.sum()),
        target_sites=int(target.sum()),best_rank=rank,mrr=1/rank if rank else 0.,
        **{'top'+str(k):bool(rank is not None and rank<=k) for k in (1,5,15)})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('selection','forward','annotations','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    seal=json.loads((a.forward/'prediction_seal.json').read_text());assert seal['complete'] and seal['predictions_sha256']==file_sha256(a.forward/'predictions_public.pt')
    selection=json.loads((a.selection/'manifest.json').read_text());assert selection['annotation_selection_sha256']==file_sha256(a.annotations)
    queries=torch.load(a.selection/'inputs_public.pt',weights_only=True);predictions=torch.load(a.forward/'predictions_public.pt',weights_only=True)
    annotations={x['frame_key']:x for x in json.loads(a.annotations.read_text()) if x['boxes_px']};assert len(annotations)==27
    # Private registered Goal labels are opened only after the public seal check.
    truth={}
    for obs in {q['observations'] for q in queries.values()}:
        for x in map(json.loads,(Path(obs)/'labels_private.jsonl').read_text().splitlines()):
            if x.get('frame_key') in queries:truth[x['frame_key']]=x
    methods=list(next(iter(predictions.values())));rows=[]
    for key,per_method in predictions.items():
        annotation=annotations[key];q=queries[key];validate_weak_box_record(annotation,q['frame']['image_sha256'])
        target=torch.tensor(truth[key]['target_world'][:2]);world=q['global_world'];base_ne=float((world[q['prior_logits'].argmax()]-target).norm())
        scored={}
        for method,pred in per_method.items():
            choice=int(pred['logprob'].argmax());ne=float((world[choice]-target).norm())
            scored[method]=dict(goal_ne_m=ne,goal_success=ne<=20,goal_world=world[choice].tolist(),
                proxies={score:{str(delta):ranking_proxy(pred,annotation['boxes_px'][0],score,delta) for delta in (-2,0,2)} for score in ('cosine','visual_score')})
        rows.append(dict(frame_key=key,task_key=q['frame']['task_key'],map_name=q['frame']['map_name'],a0_ne_m=base_ne,a0_success=base_ne<=20,methods=scored))
    common={row['frame_key'] for row in rows if all(row['methods'][m]['proxies']['cosine']['0']['candidate_present'] for m in methods)}
    summary=dict(cases=len(rows),maps=len({x['map_name'] for x in rows}),common_center_subset=len(common),
        a0=dict(correct=sum(x['a0_success'] for x in rows),ne_m=float(np.mean([x['a0_ne_m'] for x in rows]))),methods={})
    for method in methods:
        values=[x['methods'][method] for x in rows];baseline=[x['methods']['Native144'] for x in rows]
        result=dict(correct=sum(x['goal_success'] for x in values),goal_r1=float(np.mean([x['goal_success'] for x in values])),goal_ne_m=float(np.mean([x['goal_ne_m'] for x in values])),
            corrected_a0=sum(x['goal_success'] and not r['a0_success'] for x,r in zip(values,rows)),harmed_a0=sum(not x['goal_success'] and r['a0_success'] for x,r in zip(values,rows)),
            corrected_144=sum(x['goal_success'] and not b['goal_success'] for x,b in zip(values,baseline)),harmed_144=sum(not x['goal_success'] and b['goal_success'] for x,b in zip(values,baseline)),proxies={})
        for score in ('cosine','visual_score'):
            result['proxies'][score]={}
            for delta in (-2,0,2):
                proxies=[x['proxies'][score][str(delta)] for x in values];present=[x for x in proxies if x['candidate_present']]
                common_p=[x for x,r in zip(proxies,rows) if r['frame_key'] in common]
                result['proxies'][score][str(delta)]=dict(present=len(present),raw_center_support=sum(x['raw_center_support'] for x in proxies),
                    **{'top'+str(k):sum(x['top'+str(k)] for x in proxies) for k in (1,5,15)},
                    conditional_top1=sum(x['top1'] for x in present)/max(1,len(present)),mrr=float(np.mean([x['mrr'] for x in proxies])),
                    common_top1=sum(x['top1'] for x in common_p),common_top5=sum(x['top5'] for x in common_p))
        summary['methods'][method]=result
    runtime=json.loads((a.forward/'runtime.json').read_text());summary['runtime']=dict(
        medians={key:float(np.median([r[key] for r in runtime])) for key in ('preprocess_seconds','encoder_seconds','pool144_seconds','pool576_seconds')},
        head_and_export_median={m:float(np.median([r['head_seconds'][m] for r in runtime])) for m in methods},
        peak_gpu_gib=max(r['peak_gpu_gib'] for r in runtime),pooling_shape_min_cosine=min(r['pooling_shape_cosine_min'] for r in runtime))
    summary['warning']='Annotation-selected TRAIN27 AI weak appearance boxes only. Proxies are lattice-center ranking, NOT detector recall, verified instance accuracy or navigation SR.'
    summary['sha256']={k:file_sha256(v) for k,v in dict(predictions=a.forward/'predictions_public.pt',seal=a.forward/'prediction_seal.json',annotations=a.annotations,producer=Path(__file__)).items()}
    a.output.mkdir(parents=True);atomic_json(rows,a.output/'records_private.json');atomic_json(summary,a.output/'summary.json');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
