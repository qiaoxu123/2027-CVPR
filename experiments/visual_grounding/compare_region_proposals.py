"""Cached comparison on exactly the same proposal-covered candidate subsets.

Does not instantiate a model. RegionCLIP source-frame GT-box features, common
context ROI crops, and annotation-associated proposal features differ in their
visual processing. This comparison cannot isolate a pure region-architecture
benefit; pretraining, backbone and preprocessing are confounded.
"""
import argparse, collections, csv, itertools, json
from pathlib import Path
import numpy as np
import torch
from experiments.visual_grounding.freeze_dataset import read,sha
from experiments.visual_grounding.statistics import intervals

MODELS=['remoteclip_vitl14','remoteclip_vitb32','regionclip_rn50','siglip2_large384']


def main():
    p=argparse.ArgumentParser();p.add_argument('--frozen',required=True);p.add_argument('--benchmark-root',required=True);p.add_argument('--proposals',required=True);p.add_argument('--output');p.add_argument('--split',choices=['development','unseen','all'],default='development');a=p.parse_args()
    frozen=Path(a.frozen);root=Path(a.benchmark_root);proposals=Path(a.proposals);out=Path(a.output) if a.output else proposals;out.mkdir(parents=True,exist_ok=True)
    lock=json.loads((frozen/'frozen.json').read_text())
    for name in ['index','queries','attributes','retrieval']:
        if sha(frozen/f'{name}.jsonl')!=lock['files'][name]:raise ValueError('Frozen file changed')
    fingerprint=sha(frozen/'frozen.json');index=read(frozen/'index.jsonl');lookup={r['sample_id']:i for i,r in enumerate(index)};queries={q['query_sample_id']:q for q in read(frozen/'queries.jsonl')}
    subsets=read(proposals/f'proposal_common_subsets_{a.split}.jsonl');proposal_records=read(proposals/f'proposal_matching_{a.split}.jsonl');proposal_metrics=json.loads((proposals/f'proposal_metrics_{a.split}.json').read_text())
    if proposal_metrics['frozen_sha256']!=fingerprint:raise ValueError('Proposal fingerprint mismatch')
    per_model={};processing={}
    for model in MODELS:
        cache=root/model/('seen' if a.split=='development' else 'unseen')/'embeddings.pt';state=torch.load(cache,map_location='cpu',weights_only=True)
        if state['protocol']['frozen_sha256']!=fingerprint or len(state['images'])!=len(index):raise ValueError('Cached model fingerprint mismatch')
        images=state['images'];text_lookup={s:i for i,s in enumerate(state['strings'])};language=state['texts'];rows=[]
        if not torch.isfinite(images).all() or not torch.isfinite(language).all():raise ValueError('Nonfinite cached features')
        processing[model]=state['protocol']['region_source']
        for subset in subsets:
            query=queries[subset['query_id']];ids=subset['candidate_sample_ids'];positive=set(subset['positive_sample_ids']);negative=set(subset['negative_sample_ids']);matrix=images[[lookup[s] for s in ids]]
            if not set(ids)<=set(query['candidate_sample_ids']):raise ValueError('Candidate subset not from common frozen query')
            for mode in ['instruction','attribute_phrase']:
                scores=matrix@language[text_lookup[query[mode]]];order=scores.argsort(descending=True).tolist();ranking=[ids[i] for i in order];pi=[i for i,s in enumerate(ids) if s in positive];ni=[i for i,s in enumerate(ids) if s in negative]
                rows.append(dict(model=model,query_id=subset['query_id'],mode=mode,split=query['split'],map_name=query['map_name'],object_id=index[lookup[query['query_sample_id']]]['object_id'],attribute_type=query['attribute_type'],candidate_size=len(ids),r1=ranking[0] in positive,r5=bool(set(ranking[:5])&positive),pairwise=float((scores[pi,None]>scores[ni][None]).float().mean()),ranking=ranking,scores=dict(zip(ids,scores.tolist())),label_status=query['label_status']))
        per_model[model]=rows
    bmodel='regionclip_rn50_proposal_annotation_upper_bound';rows=[]
    for row in proposal_records:
        query=queries[row['query_id']];rows.append(dict(row,model=bmodel,object_id=index[lookup[query['query_sample_id']]]['object_id']))
    per_model[bmodel]=rows;processing[bmodel]='actual RPN top100 + weak-GT best-IoU annotation association; diagnostic upper bound'
    ordered_keys=sorted((s['query_id'],mode) for s in subsets for mode in ['instruction','attribute_phrase'])
    for model,rows in per_model.items():
        keyed={(r['query_id'],r['mode']):r for r in rows}
        if set(keyed)!=set(ordered_keys):raise ValueError('Model query pairing differs')
        per_model[model]=[keyed[k] for k in ordered_keys]
    summary={};flat=[]
    for model,rows in per_model.items():
        summary[model]={}
        for mode in ['instruction','attribute_phrase']:
            sub=[r for r in rows if r['mode']==mode];objects={(r['map_name'],r['object_id']) for r in sub};maps=[r['map_name'] for r in sub]
            metrics={k:intervals([r[k] for r in sub],maps) for k in ['r1','r5','pairwise']}
            bytype={kind:{k:intervals([r[k] for r in sub if r['attribute_type']==kind],[r['map_name'] for r in sub if r['attribute_type']==kind]) for k in ['r1','r5','pairwise']} for kind in sorted({r['attribute_type'] for r in sub})}
            summary[model][mode]=dict(metrics=metrics,queries=len(sub),independent_objects=len(objects),maps=len(set(maps)),candidate_size_mean=float(np.mean([r['candidate_size'] for r in sub])) if sub else None,candidate_size_range=[min(r['candidate_size'] for r in sub),max(r['candidate_size'] for r in sub)] if sub else None,by_attribute=bytype)
            for metric,values in metrics.items():flat.append(dict(comparison_type='model_metric',first_model=model,second_model='',model=model,mode=mode,metric=metric,queries=len(sub),independent_objects=len(objects),maps=len(set(maps)),mean=values['mean'],iid_ci_low=values['iid_95ci'][0] if sub else None,iid_ci_high=values['iid_95ci'][1] if sub else None,map_ci_low=values['map_cluster_95ci'][0] if sub else None,map_ci_high=values['map_cluster_95ci'][1] if sub else None))
    paired=[]
    for first,second in itertools.combinations(per_model,2):
        for mode in ['instruction','attribute_phrase']:
            left=[r for r in per_model[first] if r['mode']==mode];right=[r for r in per_model[second] if r['mode']==mode]
            if [(r['query_id'],r['mode']) for r in left]!=[(r['query_id'],r['mode']) for r in right]:raise ValueError('Pair order differs')
            for metric in ['r1','r5','pairwise']:
                differences=[float(x[metric])-float(y[metric]) for x,y in zip(left,right)]
                paired.append(dict(first=first,second=second,mode=mode,metric=metric,delta_first_minus_second=intervals(differences,[r['map_name'] for r in left]),interpretation='same-query paired bootstrap with identical candidates and language; architectural/pretraining/preprocessing confounding remains'))
    for row in paired:
        v=row['delta_first_minus_second'];mode=row['mode'];reference=summary[row['first']][mode]
        flat.append(dict(comparison_type='paired_delta_first_minus_second',first_model=row['first'],second_model=row['second'],model=row['first']+' minus '+row['second'],mode=mode,metric=row['metric'],queries=v['n'],independent_objects=reference['independent_objects'],maps=v['maps'],mean=v['mean'],iid_ci_low=v['iid_95ci'][0] if v['n'] else None,iid_ci_high=v['iid_95ci'][1] if v['n'] else None,map_ci_low=v['map_cluster_95ci'][0] if v['n'] else None,map_ci_high=v['map_cluster_95ci'][1] if v['n'] else None))
    candidate_size_distribution=dict(collections.Counter(len(s['candidate_sample_ids']) for s in subsets))
    r5_guaranteed_by_size=sum(len(s['candidate_sample_ids'])<=5 for s in subsets)
    result=dict(candidate_size_distribution=candidate_size_distribution,r5_guaranteed_at_least_queries=r5_guaranteed_by_size,frozen_sha256=fingerprint,split=a.split,source='existing cached embeddings and existing proposal predictions; no new model inference',query_count=len(subsets),input_query_count=proposal_metrics['input_queries'],coverage_fraction=len(subsets)/proposal_metrics['input_queries'] if proposal_metrics['input_queries'] else None,visual_processing=processing,summary=summary,paired_comparisons=paired,limitations=['Labels are linguistic weak supervision, not independently human-verified visual attributes.','RPN candidate coverage and best-IoU weak-GT association select a small diagnostic upper-bound subset.','Native RegionCLIP source-frame bbox features and common-context letterboxed ROI crop encoders have backbone, pretraining and image-processing confounds; any win cannot establish a pure region-architecture benefit.','R@5 is mechanically saturated when all candidates number at most five.','Multiple queries from the same object are correlated; independent object counts and map-cluster intervals are reported.'])
    (out/'matched_comparison.json').write_text(json.dumps(result,indent=2))
    with (out/'matched_comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]) if flat else ['model','mode','metric']);writer.writeheader();writer.writerows(flat)
    (out/'matched_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for rows in per_model.values() for r in rows))
    print(json.dumps(dict(query_count=len(subsets),summary={m:{mode:{k:v['mean'] for k,v in d['metrics'].items()} for mode,d in modes.items()} for m,modes in summary.items()})),flush=True)

if __name__=='__main__':main()
