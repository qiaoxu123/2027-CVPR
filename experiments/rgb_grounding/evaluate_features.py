"""Frozen representation diagnostics on identical judged multi-positive ROIs."""
import argparse
import json
from pathlib import Path
import torch
from experiments.bidirectional_core.stats import summarize_rows,paired_difference
from graphnav.checkpointing import file_sha256


def load(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    if not manifest['complete']:raise ValueError('incomplete features')
    rows=[]
    for p in manifest['chunks']:
        if file_sha256(root/p)!=manifest['files'][p]:raise ValueError('changed cached features')
        rows.extend(torch.load(root/p,map_location='cpu',weights_only=True))
    return rows


def metrics(scores,row):
    pos=row['labels']['positive_roi_mask'].bool();judged=row['labels']['semantic_valid_mask'].bool();valid=row['public']['roi_valid'].bool();m=row['metadata']
    objects=m['roi_objects'];positive=pos & valid;eligible=judged & valid
    if not positive.any() or not (eligible & ~positive).any():
        return {key:None for key in ['r1','r4','r8','tie_aware_r1','top_tie','candidate_count','forward_r1','reverse_r1','building_hard_r1']}
    grouped={}
    for j,region in enumerate(objects):
        if eligible[j]:grouped.setdefault(region['object_id'],[]).append(j)
    ranked=sorted(grouped,key=lambda oid:(-float(scores[grouped[oid]].mean()),oid))
    positives={objects[j]['object_id'] for j in range(len(objects)) if positive[j]}
    top=ranked[0];best=float(scores[grouped[top]].mean());tied=[oid for oid in ranked if abs(float(scores[grouped[oid]].mean())-best)<1e-7]
    return dict(r1=float(top in positives),r4=float(any(x in positives for x in ranked[:4])),r8=float(any(x in positives for x in ranked[:8])),
        tie_aware_r1=sum(x in positives for x in tied)/len(tied),top_tie=float(len(tied)>1),candidate_count=len(ranked),
        forward_r1=float(top in positives) if m['direction']=='forward' else None,
        reverse_r1=float(top in positives) if m['direction']=='reverse' else None,
        building_hard_r1=float(top in positives) if all(objects[j]['object_type']=='Building' for j in positive.nonzero().ravel()) else None)


def run(args):
    output=Path(args.output)
    if output.exists():raise ValueError('refusing metric overwrite')
    output.mkdir(parents=True);rows=load(args.features)
    records={name+'_'+language:[] for name in rows[0]['features'] for language in ['full','phrase']}
    for row in rows:
        meta=row['metadata']
        for representation,images in row['features'].items():
            for language,text in row['texts'].items():
                scores=images.float()@text.float()
                result=dict(sample_id=meta['sample_id'],family_id=meta['family_id'],map_name=meta['map_name'],direction=meta['direction'],
                    phrase_supported=row['phrase_supported'],metrics=metrics(scores,row),scores=scores.tolist())
                records[representation+'_'+language].append(result)
    summary={}
    for name,predictions in records.items():
        summary[name]={key:summarize_rows(predictions,key,repetitions=2000) for key in predictions[0]['metrics']}
        with (output/(name+'_predictions.jsonl')).open('w') as h:
            for result in predictions:h.write(json.dumps(result)+'\n')
    paired={name:{key:paired_difference(records['patch_pool_full'],predictions,key,repetitions=2000) for key in ['r1','forward_r1','reverse_r1','building_hard_r1']} for name,predictions in records.items()}
    (output/'metrics.json').write_text(json.dumps(dict(models=summary,paired_vs_patch_full=paired,features_manifest_sha256=file_sha256(Path(args.features)/'manifest.json'),annotation_status='all CityNav language/geometry supervision weak, not human visual verification'),indent=2))
    print(json.dumps({name:{k:v[k]['mean'] for k in ['r1','forward_r1','reverse_r1']} for name,v in summary.items()},indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--output',required=True);run(p.parse_args())


if __name__=='__main__':main()
