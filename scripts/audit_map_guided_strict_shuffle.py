"""Post-hoc control audit: same-caption target-only permutation, no new network."""
import argparse,hashlib,json
from pathlib import Path
import cv2,numpy as np,torch
from transformers import AutoTokenizer
from graphnav.checkpointing import atomic_json
from graphnav.map_guided_grounding import (public_landmarks,project_polygon,reference_crop_bounds,crop_reference,
    reference_caption,reference_candidates,target_word_mask,geometry_scores,score_candidates)
from graphnav.object_query_belief import ObjectQueryRanker
from scripts.train_object_query_belief import load
from scripts.diagnose_map_guided_visual_grounding import TOKENIZER,summarize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['records','cache','catalog','reference-cache','native-checkpoint','output']:p.add_argument('--'+n,required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise ValueError('No overwrite')
    meta,rows=load(a.cache)
    if meta['partition']!='dev':raise ValueError('This audit uses only the frozen internal Dev')
    lookup={(r['episode'],r['step']):r for r in rows}
    records=[json.loads(line) for line in Path(a.records).read_text().splitlines()]
    catalog=json.loads(Path(a.catalog).read_text());named={m:public_landmarks(catalog,m) for m in {r['map'] for r in records}}
    relations_cache={};tokenizer=AutoTokenizer.from_pretrained(TOKENIZER,local_files_only=True,use_fast=True)
    native=ObjectQueryRanker().cuda();checkpoint=torch.load(a.native_checkpoint,map_location='cpu',weights_only=True);native.load_state_dict(checkpoint['model']);native.eval().requires_grad_(False)
    out.mkdir();out_records=[]
    with torch.inference_mode():
        for number,record in enumerate(records):
            row=lookup[(record['episode'],record['step'])];frame=torch.load(record['raw_frame'],weights_only=True)
            bgr=cv2.imread(str(Path(record['raw_frame']).with_suffix('.png')));landmarks=named[record['map']]
            key=(record['map'],row['instruction'])
            if key not in relations_cache:
                entities,relations,_,_=reference_candidates(landmarks,[dict(valid_roi=False) for _ in landmarks],row['instruction'])
                relations_cache[key]=(entities,relations)
            entities,relations=relations_cache[key]
            indices=[next(i for i,e in enumerate(landmarks) if e['key']==identifier) for identifier in record['reference_ids']]
            refs=[];languages=[]
            for i in indices:
                projection=project_polygon(landmarks[i]['geometry_m'],frame['footprint_world'].numpy(),bgr.shape,frame['depth_valid'].numpy())
                crop=crop_reference(bgr,projection);x0,y0,x1,y1=reference_crop_bounds(bgr.shape,projection)
                depth=frame['depth_valid'].numpy()[y0:y1,x0:x1].copy();caption=reference_caption(landmarks[i])
                digest=hashlib.sha256(crop.tobytes()+depth.tobytes()+caption.encode()).hexdigest()
                cached=Path(a.reference_cache)/(digest+'.pt')
                if not cached.is_file():raise ValueError('Audit must not infer/rebuild missing reference features')
                feature=torch.load(cached,weights_only=True);refs.append(feature['feature']);languages.append(feature['language'])
            refs=torch.stack(refs) if refs else torch.empty(0,256)
            languages=torch.stack(languages) if languages else torch.empty(0,256)
            valid=row['valid'];features=row['features'].float().clone();ids=valid.nonzero().flatten()
            features[ids]=features[ids.flip(0)]
            mask=target_word_mask(row['instruction'],tokenizer,len(row['text']),row['text_mask'])
            language=row['text'].float()[mask].mean(0) if mask.any() else None
            centers=row['query_world'].float();prior=row['base_logits'][torch.cdist(row['grid_world'].float(),centers).argmin(0)]
            prior,fields=geometry_scores(centers.numpy(),prior,entities,relations)
            _,scores,_=score_candidates(prior,fields,relations,features,language,refs,languages,indices,valid)
            grid=row['grid_world'].float()[None].cuda();kernel=torch.exp(-torch.cdist(grid,centers[None].cuda()).square()/(2*8.**2))*row['observed'][None,:,None].cuda()*valid[None,None].cuda()
            belief=native.update_belief(row['base_logits'][None].cuda(),scores[None].cuda(),torch.tensor([-10.],device='cuda'),kernel,row['observed'][None].cuda(),valid[None].cuda(),bound=2.)[0][0].cpu()
            # Evaluation begins only after the fixed inference output above.
            distance=(row['grid_world'].float()-row['labels']['target_world'][:2]).norm(dim=-1)
            record['methods']['joint_shuffled_rgb']=record['methods']['shuffled_rgb']
            record['methods']['shuffled_rgb']=dict(error=float(distance[belief.argmax()]),rank=int(scores.argmax()))
            original=record['instance']['native']
            instance=dict(original)
            if instance['eligible']:
                iou=torch.tensor(instance['iou']);order=scores.argsort(descending=True)
                for k,n in [('top1',1),('top5',5),('top15',15)]:instance[k]=bool(iou[order[:n]].max()>=.5)
            record['instance']['joint_shuffled_rgb']=record['instance']['shuffled_rgb'];record['instance']['shuffled_rgb']=instance
            out_records.append(record)
            if (number+1)%100==0:print(number+1,flush=True)
    with (out/'records.jsonl').open('w') as f:
        for r in out_records:f.write(json.dumps(r)+'\n')
    atomic_json(summarize(out_records),out/'summary.json')
    atomic_json(dict(complete=True,post_hoc=True,no_training=True,no_network_forward=True,
                     control='same-caption target query feature reversal only; reference features/geometry/scoring unchanged',
                     records=len(records)),out/'complete.json')


if __name__=='__main__':main()
