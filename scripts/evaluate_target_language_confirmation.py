"""One-shot TRAIN confirmation: seal all public predictions before GT scoring."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
import torch
import numpy as np
from graphnav.checkpointing import atomic_json,atomic_save,file_sha256
from graphnav.local_chain_raster import rasterized_goal_logprob
from graphnav.visual_chain_position import WorldLikelihoodPositionHead
from scripts.train_local_chain_position import INPUTS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('cache','roles','labels','selection','suite','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('one-shot confirmation; preserve previous outcomes')
    torch.set_num_threads(4);started=time.perf_counter()
    meta=json.loads((a.cache/'manifest.json').read_text())
    freeze=json.loads((a.selection/'selection_freeze.json').read_text())
    if not meta['complete'] or meta['split']!='train_seen' or set(meta['map_assignment'].values())!={'confirm'}:
        raise ValueError('only complete held-out TRAIN maps allowed')
    full=a.suite/'full_instruction_s0/final.pt';role=a.suite/'role_s0/final.pt'
    if file_sha256(full)!=freeze['local_checkpoint_sha256']:raise ValueError('model changed after selection')
    config=json.loads((a.suite/'full_instruction_s0/config.json').read_text())['config']
    source_manifest=json.loads((Path(config['cache'])/'manifest.json').read_text())
    for key in ('checkpoint_sha256','public_catalog_sha256','static_queries_sha256'):
        if meta[key]!=source_manifest[key]:raise ValueError('source model or public dataset changed')
    vectors=torch.load(a.roles,map_location='cpu',weights_only=True);rows=[]
    for chunk in meta['chunks']:
        path=a.cache/chunk['file']
        if file_sha256(path)!=chunk['sha256']:raise ValueError('corrupt confirmation features')
        for row in torch.load(path,map_location='cpu',weights_only=True)['records']:
            assert row['subset']=='confirm'
            copied=dict(row);copied['full_text']=row['context_text'];copied['context_text']=vectors[row['task_key']]['reference_vector']
            rows.append(copied)
    expected={f'train_{i:05d}' for i in freeze['indices']}
    if {r['task_key'] for r in rows}!=expected:raise ValueError('missing selected targets; do not silently filter')
    a.output.mkdir(parents=True)
    models={'FrozenV3':(Path(config['warm_start']),'role','real'),
            'TrainedRole':(role,'role','real'),
            'TrainedFullTarget':(full,'full-instruction','real'),
            'FullTargetShuffleRGB':(full,'full-instruction','shuffle')}
    predictions={};posteriors={}
    for name,(checkpoint,language,control) in models.items():
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if name!='FrozenV3' and state['config']['target_language']!=language:raise ValueError('trained language interface mismatch')
        model=WorldLikelihoodPositionHead().cuda().eval().requires_grad_(False);model.load_state_dict(state['model'],strict=True)
        outputs=[];probabilities=[]
        with torch.inference_mode():
            for start in range(0,len(rows),8):
                batch=rows[start:start+8]
                x={k:torch.stack([r[k] for r in batch]).cuda() for k in INPUTS}
                if language=='full-instruction':x['target_text']=torch.stack([r['full_text'] for r in batch]).cuda()
                if control=='shuffle':
                    image=x['image_features'].clone()
                    for i in range(len(batch)):
                        ids=x['local_valid'][i].nonzero().flatten()
                        image[i,ids]=x['image_features'][i,ids.flip(0)]
                    x['image_features']=image
                packed=model(**x)
                logits=rasterized_goal_logprob(packed,x['local_uv'],x['local_valid'],global_cells=x['prior_logits'].shape[-1])
                posterior=logits.softmax(-1).cpu();probabilities.append(posterior)
                for i,row in enumerate(batch):
                    choice=int(posterior[i].argmax());base=int(row['prior_logits'].argmax())
                    outputs.append(dict(frame_key=row['frame_key'],task_key=row['task_key'],
                        predicted_world=row['candidate_world'][choice].tolist(),
                        prior_world=row['candidate_world'][base].tolist(),chosen_index=choice))
        predictions[name]=outputs;posteriors[name]=torch.cat(probabilities)
    # Explicit boundary: all predictions/weights are sealed before opening labels.
    atomic_json(predictions,a.output/'predictions_public.json')
    atomic_save(posteriors,a.output/'posteriors_public.pt')
    atomic_json(dict(predictions_sha256=file_sha256(a.output/'predictions_public.json'),
        posteriors_sha256=file_sha256(a.output/'posteriors_public.pt'),
        checkpoint_sha256={n:file_sha256(c) for n,(c,_,_) in models.items()},
        source_sha256=file_sha256(__file__),selection_sha256=file_sha256(a.selection/'selection_freeze.json'),
        feature_manifest_sha256=file_sha256(a.cache/'manifest.json'),reference_vectors_sha256=file_sha256(a.roles),
        GT_read_before_seal=False,states=len(rows),targets=len(expected)),a.output/'prediction_seal.json')
    truth={r['frame_key']:r for r in map(json.loads,a.labels.read_text().splitlines()) if r['status']=='captured'}
    results={}
    for name,outputs in predictions.items():
        scored=[]
        for i,out in enumerate(outputs):
            gt=truth[out['frame_key']];target=np.asarray(gt['target_world'][:2]);ne=float(np.linalg.norm(np.asarray(out['predicted_world'])-target));base_ne=float(np.linalg.norm(np.asarray(out['prior_world'])-target))
            mask=(rows[i]['candidate_world'][:784]-torch.tensor(target).float()).norm(dim=-1)<=20
            scored.append(dict(**out,category=gt['category'],ne_m=ne,success=ne<=20,baseline_ne_m=base_ne,baseline_success=base_ne<=20,
                target_20m_probability_mass=float(posteriors[name][i,mask].sum())))
        by_task=defaultdict(list)
        for x in scored:by_task[x['task_key']].append(x)
        results[name]=dict(metrics=dict(frames=len(scored),physical_targets=len(by_task),goal_r1=float(np.mean([x['success'] for x in scored])),
            goal_ne_m=float(np.mean([x['ne_m'] for x in scored])),target_balanced_r1=float(np.mean([np.mean([r['success'] for r in rs]) for rs in by_task.values()])),
            probability_mass_20m=float(np.mean([x['target_20m_probability_mass'] for x in scored])),
            corrections=sum(x['success'] and not x['baseline_success'] for x in scored),harms=sum(not x['success'] and x['baseline_success'] for x in scored)),rows=scored)
    ours=results['TrainedFullTarget']['metrics']
    gate=all(ours['goal_r1']>results[n]['metrics']['goal_r1'] and ours['goal_ne_m']<results[n]['metrics']['goal_ne_m'] for n in ('FrozenV3','TrainedRole','FullTargetShuffleRGB'))
    paired={}
    for name in ('FrozenV3','TrainedRole','FullTargetShuffleRGB'):
        pairs=list(zip(results['TrainedFullTarget']['rows'],results[name]['rows']))
        task_deltas=defaultdict(list)
        for new,old in pairs:
            assert new['frame_key']==old['frame_key']
            task_deltas[new['task_key']].append(int(new['success'])-int(old['success']))
        values=np.array([sum(v) for v in task_deltas.values()]);counts=np.array([len(v) for v in task_deltas.values()]);ids=np.random.default_rng(0).integers(0,len(values),(10000,len(values)))
        ci=np.quantile(values[ids].sum(-1)/counts[ids].sum(-1),[.025,.975])
        paired[name]=dict(corrected=[n['frame_key'] for n,o in pairs if n['success'] and not o['success']],harmed=[n['frame_key'] for n,o in pairs if not n['success'] and o['success']],target_group_bootstrap95_r1_delta=ci.tolist())
    atomic_json(dict(results=results,paired=paired,navigation_gate=gate,seconds=time.perf_counter()-started,
        warning='One-shot TRAIN confirmation; weak world-location evaluation, not pixel-instance matching or navigation SR'),a.output/'results.json')
    print(json.dumps(dict(navigation_gate=gate,metrics={n:r['metrics'] for n,r in results.items()})),flush=True)


if __name__=='__main__':main()
