"""All supported real-candidate attribute exchanges; weak labels explicit."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import torch
from .city_features import CropEncoder
from .evaluate_features import load,metrics
from .goal_evidence import load_head,encode_scores
from experiments.bidirectional_core.cache import conservative_attributes
from experiments.bidirectional_core.stats import summarize_rows
from graphnav.checkpointing import file_sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);args=p.parse_args();root=Path(args.output)
    if root.exists():raise ValueError('refusing counterfactual overwrite')
    root.mkdir(parents=True);original=load(args.features);queries=[];audit=[];encoder=CropEncoder()
    for row in original:
        m=row['metadata'];regions=m['roi_objects'];pos=row['labels']['positive_roi_mask'];classes={r['object_type'] for r,p in zip(regions,pos) if p}
        if len(classes)!=1:continue
        cls=next(iter(classes));attrs=conservative_attributes(row['phrase'],cls)
        for kind,value in attrs.items():
            values=sorted({r.get('attributes',{}).get(kind,{}).get('label') for r in regions if r['object_type']==cls and r.get('attributes',{}).get(kind,{}).get('label') not in [None,value]})
            for replacement in values:
                swapped=dict(attrs);swapped[kind]=replacement;positive=[];judged=[]
                for r,v in zip(regions,row['public']['roi_valid']):
                    candidate=r.get('attributes',{});shared=[k for k in swapped if k in candidate]
                    same=bool(v and r['object_type']==cls and len(shared)==len(swapped) and all(candidate[k]['label']==swapped[k] for k in shared))
                    contrary=bool(v and r['object_type']==cls and any(candidate[k]['label']!=swapped[k] for k in shared))
                    positive.append(same);judged.append(same or contrary)
                if not any(positive) or not any(v and not q for v,q in zip(judged,positive)):continue
                # Only the bound visual phrase is exchanged. No inference access
                # to object IDs/attributes; those choose the evaluator's cohort.
                phrase=row['phrase'].replace(value,replacement,1)
                if phrase==row['phrase']:continue
                query=dict(row);query['metadata']=dict(m,sample_id=m['sample_id']+':cf:'+hashlib.sha256((kind+replacement).encode()).hexdigest()[:8]);query['texts']={'phrase':encoder.native.texts([phrase]).cpu().half()[0]};query['phrase']=phrase
                query['labels']=dict(positive_roi_mask=torch.tensor(positive),semantic_valid_mask=torch.tensor(judged));queries.append(query);audit.append(dict(sample_id=query['metadata']['sample_id'],original_sample_id=m['sample_id'],attribute=kind,original_phrase=row['phrase'],counterfactual_phrase=phrase,positive_instances=sum(positive),known_candidate_count=sum(judged)))
    model=load_head(args.checkpoint,'cuda');predictions={}
    if model.token_attention:raise ValueError('this isolated phrase test requires projection head, not inconsistent unchanged word tokens')
    for name,head in [('native',None),('aerial_city_projection',model)]:
        scores=encode_scores(queries,head,{},'real','cuda');rows=[]
        for query in queries:
            m=query['metadata'];score=scores[m['sample_id']]['score'][0];rows.append(dict(sample_id=m['sample_id'],family_id=m['family_id'],map_name=m['map_name'],direction=m['direction'],metrics=metrics(score,query),scores=score.tolist(),phrase=query['phrase']))
        predictions[name]=rows
        with (root/(name+'_predictions.jsonl')).open('w') as h:
            for r in rows:h.write(json.dumps(r)+'\n')
    result=dict(queries=len(queries),original_states=len({r['original_sample_id'] for r in audit}),swaps=audit,models={name:summarize_rows(rows,'r1',repetitions=2000) for name,rows in predictions.items()},features_manifest_sha256=file_sha256(Path(args.features)/'manifest.json'),checkpoint_sha256=file_sha256(args.checkpoint),supervision='existing consistent language-derived object attributes only, not human visual truth; opposite attributes required among actually observed valid ROIs; missing/unknown never negative',purpose='fixed all-supported diagnostic, not Seen prompt/threshold tuning; no counterfactual label is added to training')
    (root/'metrics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
