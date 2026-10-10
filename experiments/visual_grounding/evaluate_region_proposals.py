"""Official RegionCLIP RPN coverage and label-associated diagnostic matching.

Proposal generation uses only observed RGB. Best-IoU association with weak
geometric target boxes is an evaluation annotation operation, so feature
matching is a diagnostic upper bound on localization, not an unsupervised
proposal-to-object inference result. Missing proposals remain Unknown.
"""
import argparse, collections, hashlib, json, time
from pathlib import Path
import cv2
import torch
from torchvision.ops import box_iou
from experiments.visual_grounding.adapters.regionclip import RegionCLIPEncoder
from experiments.visual_grounding.freeze_dataset import read, sha
from experiments.visual_grounding.statistics import intervals


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--frozen',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--official-repo',required=True)
    parser.add_argument('--split',choices=['development','unseen','all'],default='development')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--batch-size',type=int,default=16)
    args=parser.parse_args()
    frozen=Path(args.frozen);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    result_file=out/f'proposal_metrics_{args.split}.json'
    if result_file.exists():raise ValueError('Completed proposal evaluation exists; refusing repeat')
    lock=json.loads((frozen/'frozen.json').read_text())
    for name in ['index','queries','retrieval','attributes']:
        if sha(frozen/f'{name}.jsonl')!=lock['files'][name]:raise ValueError('Frozen data changed')
    all_index=read(frozen/'index.jsonl')
    index=[r for r in all_index if args.split=='all' or (r['split']=='val_unseen')==(args.split=='unseen')]
    encoder=RegionCLIPEncoder(args.checkpoint,args.official_repo,args.device,args.batch_size)
    encoder.enable_proposals(args.checkpoint,topk=100)
    byframe=collections.defaultdict(list)
    for row in index:byframe[row['roi_source']].append(row)
    vectors={};coverage={};predictions=[];frame_predictions=[];errors=[]
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with (out/f'proposal_predictions_{args.split}.jsonl').open('w') as stream:
        for n,(frame,records) in enumerate(byframe.items()):
            image=cv2.imread(frame)
            if image is None:raise FileNotFoundError(frame)
            boxes,objectness=encoder.proposals(image)
            if not torch.isfinite(boxes).all() or not torch.isfinite(objectness).all():raise ValueError('Nonfinite RPN outputs')
            # Output is official post-NMS strongest-objectness top100, independent of GT.
            frame_predictions.append(dict(roi_source=frame,num_proposals=len(boxes),proposal_boxes=boxes.tolist(),objectness_logits=objectness.tolist(),split=records[0]['split']))
            gt=torch.tensor([r['roi_bbox'] for r in records],dtype=torch.float32)
            if len(boxes):
                max_iou,best=box_iou(gt,boxes).max(1)
                used=sorted(set(best.tolist()));features=[]
                for offset in range(0,len(used),args.batch_size):
                    subset=used[offset:offset+args.batch_size]
                    features.append(encoder.regions(image,boxes[subset].tolist()))
                feature_by_proposal=dict(zip(used,torch.cat(features)))
            else:max_iou=torch.zeros(len(gt));best=torch.full((len(gt),),-1)
            for i,row in enumerate(records):
                rec=dict(sample_id=row['sample_id'],split=row['split'],map_name=row['map_name'],object_id=row['object_id'],object_type=row['object_type'],roi_source=frame,reference_weak_geometry_bbox=row['roi_bbox'],proposal_count=len(boxes),max_iou=float(max_iou[i]),proposal_index=int(best[i]),recalled_05=bool(max_iou[i]>=.5),recalled_075=bool(max_iou[i]>=.75),association='best IoU annotation-assisted diagnostic upper bound; not unsupervised object association')
                if len(boxes):rec['best_proposal_bbox']=boxes[best[i]].tolist();vectors[row['sample_id']]=feature_by_proposal[int(best[i])]
                coverage[row['sample_id']]=rec['recalled_05']
                predictions.append(rec);stream.write(json.dumps(rec)+'\n')
            if n%100==0:print(json.dumps(dict(proposal_frames=n,total_frames=len(byframe),seconds=time.perf_counter()-start)),flush=True)
    # Candidate IDs and language are frozen common benchmark data. Require a
    # covered positive and covered trustworthy weak negative; do not turn
    # unobserved or undetected candidates into negatives.
    queries=[q for q in read(frozen/'queries.jsonl') if q['query_sample_id'] in coverage]
    strings=sorted({q[k] for q in queries for k in ['instruction','attribute_phrase']})
    text_vectors=dict(zip(strings,encoder.texts(strings)))
    rankings=[];common_subsets=[]
    for query in queries:
        ids=[s for s in query['candidate_sample_ids'] if coverage.get(s,False)]
        positive=set(query['positive_sample_ids'])&set(ids);negative=set(query['negative_sample_ids'])&set(ids)
        if len(ids)<2 or not positive or not negative:continue
        common_subsets.append(dict(query_id=query['query_sample_id'],candidate_sample_ids=ids,positive_sample_ids=sorted(positive),negative_sample_ids=sorted(negative)))
        image_features=torch.stack([vectors[s] for s in ids])
        for mode in ['instruction','attribute_phrase']:
            scores=image_features@text_vectors[query[mode]];ordered=scores.argsort(descending=True).tolist();rank=[ids[i] for i in ordered];pi=[i for i,s in enumerate(ids) if s in positive];ni=[i for i,s in enumerate(ids) if s in negative]
            rankings.append(dict(query_id=query['query_sample_id'],split=query['split'],map_name=query['map_name'],attribute_type=query['attribute_type'],mode=mode,candidate_size=len(ids),r1=rank[0] in positive,r5=bool(set(rank[:5])&positive),pairwise=float((scores[pi,None]>scores[ni][None]).float().mean()),ranking=rank,scores=dict(zip(ids,scores.tolist())),label_status=query['label_status'],interpretation='Best-IoU annotation-associated proposal features; weak-language labels; diagnostic upper bound only'))
    groups={}
    for axis,key in [('split',lambda r:r['split']),('map',lambda r:r['map_name']),('type',lambda r:r['object_type'])]:
        grouped=collections.defaultdict(list)
        for rec in predictions:grouped[key(rec)].append(rec)
        groups[axis]={name:dict(regions=len(rows),independent_objects=len({(r['map_name'],r['object_id']) for r in rows}),frames=len({r['roi_source'] for r in rows}),recall_iou05=intervals([r['recalled_05'] for r in rows],[r['map_name'] for r in rows]),recall_iou075=intervals([r['recalled_075'] for r in rows],[r['map_name'] for r in rows])) for name,rows in grouped.items()}
    matching={}
    for split in sorted({r['split'] for r in rankings}):
        matching[split]={}
        for mode in ['instruction','attribute_phrase']:
            rows=[r for r in rankings if r['split']==split and r['mode']==mode]
            matching[split][mode]={k:intervals([r[k] for r in rows],[r['map_name'] for r in rows]) for k in ['r1','r5','pairwise']}
    result=dict(frozen_sha256=sha(frozen/'frozen.json'),split=args.split,config=dict(topk=100,iou_thresholds=[.5,.75],no_prompt_tuning=True),reference='Projected geometric bboxes; not reviewed semantic masks',association='Best-IoU label association is an annotation-assisted upper bound; not an unsupervised deployment result',num_regions=len(index),num_frames=len(byframe),groups=groups,matching=matching,matching_queries=len(common_subsets),input_queries=len(queries),elapsed_seconds=time.perf_counter()-start,peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),errors=errors)
    (out/f'proposal_frames_{args.split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in frame_predictions))
    (out/f'proposal_matching_{args.split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rankings))
    (out/f'proposal_common_subsets_{args.split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in common_subsets))
    torch.save(dict(sample_vectors=vectors,coverage=coverage,protocol=result['association'],frozen_sha256=result['frozen_sha256']),out/f'proposal_embeddings_{args.split}.pt')
    result_file.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
