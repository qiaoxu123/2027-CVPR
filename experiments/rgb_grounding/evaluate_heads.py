"""Paired native/pretrained/adapted semantic comparison on a frozen population."""
import argparse
import json
from pathlib import Path
import time
import torch
from .evaluate_features import load,metrics
from .goal_evidence import load_head,encode_scores
from experiments.bidirectional_core.stats import summarize_rows,paired_difference
from experiments.bidirectional_core.cache import jsonl
from graphnav.checkpointing import file_sha256


def size_group(side):return 'unknown' if side is None else 'lt32' if side<32 else '32to63' if side<64 else '64plus'


def run(args):
    output=Path(args.output)
    if output.exists():raise ValueError('refusing semantic result overwrite')
    output.mkdir(parents=True);started=time.time();rows=load(args.features)
    rows=[r for r in rows if bool(r['labels']['positive_roi_mask'].any() and (r['labels']['semantic_valid_mask']&~r['labels']['positive_roi_mask']).any())]
    controls={r['sample_id']:r for r in load(args.controls)} if args.controls else {}
    episodes={r['episode_id']:r for r in jsonl(args.episodes)} if args.episodes else {}
    heads=json.loads(Path(args.heads).read_text());predictions={};resources={}
    for name,checkpoint in heads.items():
        model=load_head(checkpoint,args.device)
        for kind in (['real','zero','shuffled'] if controls else ['real']):
            if torch.cuda.is_available():torch.cuda.synchronize()
            start=time.time();scores=encode_scores(rows,model,controls,kind,args.device)
            if torch.cuda.is_available():torch.cuda.synchronize()
            resources[name+'_'+kind]=dict(seconds=time.time()-start,states=len(rows),checkpoint_sha256=file_sha256(checkpoint) if checkpoint!='native' else None,trainable_parameters=sum(p.numel() for p in model.parameters()) if model else 0)
            records=[]
            for row in rows:
                m=row['metadata'];record=scores[m['sample_id']];positive=row['labels']['positive_roi_mask'];objects=m['roi_objects'];episode=episodes.get(m.get('episode_id'));target=episode['private_labels']['target_object_id'] if episode else None
                sizes=[r['native_min_side'] for r in objects if target is not None and r['object_id']==str(target)];side=max(sizes) if sizes else None
                primary_types={r['object_type'] for r,p in zip(objects,positive) if p};group=next(iter(primary_types)) if len(primary_types)==1 else 'multiple'
                records.append(dict(sample_id=m['sample_id'],family_id=m['family_id'],map_name=m['map_name'],direction=m['direction'],pixel_group=size_group(side),native_target_min_side=side,object_type=group,metrics=metrics(record['score'][0],row),scores=record['score'][0].tolist(),confidence=record['confidence'][0].tolist(),query=row['phrase']))
            predictions[name+'_'+kind]=records
            with (output/(name+'_'+kind+'_predictions.jsonl')).open('w') as h:
                for record in records:h.write(json.dumps(record)+'\n')
    keys=list(next(iter(predictions.values()))[0]['metrics'])
    summaries={name:{key:summarize_rows(records,key,repetitions=2000) for key in keys} for name,records in predictions.items()}
    paired={name:{key:paired_difference(predictions['native_real'],records,key,repetitions=2000) for key in ['r1','forward_r1','building_hard_r1']} for name,records in predictions.items()}
    rgb_pairs={name:{kind:{key:paired_difference(predictions[name+'_'+kind],predictions[name+'_real'],key,repetitions=2000) for key in ['r1','forward_r1','building_hard_r1']} for kind in ['zero','shuffled']} for name in heads} if controls else {}
    groups={name:{axis:{group:{key:summarize_rows([r for r in records if r[axis]==group],key,repetitions=2000) for key in ['r1','forward_r1']} for group in sorted({r[axis] for r in records})} for axis in ['pixel_group','object_type','direction']} for name,records in predictions.items()}
    result=dict(models=summaries,paired_vs_native=paired,real_minus_control=rgb_pairs,groups=groups,resources=resources,actual_semantic_states=len(rows),features_manifest_sha256=file_sha256(Path(args.features)/'manifest.json'),heads_file_sha256=file_sha256(args.heads),seconds=time.time()-started,weak_city_labels=True,unknown_candidates_excluded=True,selection='all provided checkpoints selected only on their internal Dev; this evaluator does not select checkpoints')
    (output/'metrics.json').write_text(json.dumps(result,indent=2));print(json.dumps({name:{k:r[k]['mean'] for k in ['r1','forward_r1']} for name,r in summaries.items()},indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--heads',required=True);p.add_argument('--controls');p.add_argument('--episodes');p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');run(p.parse_args())


if __name__=='__main__':main()
