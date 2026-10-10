"""Observed pixels versus occluded/invalid pixels; diagnostic, not semantic labels."""
import argparse,json,time
from pathlib import Path
import cv2
import torch
from experiments.visual_grounding.evaluate_zero_shot import load_encoder,tensor,prompt,validate
from experiments.visual_grounding.freeze_dataset import read,sha
from experiments.visual_grounding.statistics import intervals


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--checkpoint');p.add_argument('--official-repo');p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');p.add_argument('--batch-size',type=int,default=16);a=p.parse_args();data=Path(a.data);out=Path(a.output)
    dest=out/'pixel_interventions_seen.json'
    if dest.exists():raise ValueError('refusing completed diagnostic overwrite')
    state=torch.load(out/'embeddings.pt',map_location='cpu',weights_only=True);index=read(data/'index.jsonl');lookup={r['sample_id']:i for i,r in enumerate(index)};strings={s:i for i,s in enumerate(state['strings'])};attributes=[r for r in read(data/'attributes.jsonl') if r['split']=='val_seen' and r['label_status']=='language_weak_not_visual_verified']
    # Fixed first 150 object-balanced cases by sample_id; no scores select perturbations.
    rows=sorted(attributes,key=lambda r:(r['sample_id'],r['attribute_type']))[:150];encoder=load_encoder(a);pred=[];start=time.perf_counter()
    for kind in ['occluded_center','invalid_black']:
        for offset in range(0,len(rows),a.batch_size):
            chunk=rows[offset:offset+a.batch_size];images=[];boxes=[]
            for r in chunk:
                record=index[lookup[r['sample_id']]];im=cv2.imread(record['roi_source'] if a.model=='regionclip' else record['roi_path'])
                if im is None:raise FileNotFoundError(record['sample_id'])
                if a.model=='regionclip':x0,y0,x1,y1=record['roi_bbox'];boxes.append(record['roi_bbox'])
                else:h,w=im.shape[:2];x0,y0,x1,y1=0,0,w,h
                if kind=='invalid_black':im[:]=0
                else:
                    dx=int((x1-x0)*.1);dy=int((y1-y0)*.1);im[y0+dy:y1-dy,x0+dx:x1-dx]=0
                images.append(im)
            vectors=tensor(encoder.regions(images,boxes) if a.model=='regionclip' else encoder.images(images));validate(vectors)
            for r,v in zip(chunk,vectors):
                i=lookup[r['sample_id']];language=state['texts'][strings[prompt(r['attribute_type'],r['label'])]];original=float(state['images'][i]@language);changed=float(v@language)
                pred.append(dict(sample_id=r['sample_id'],map_name=r['map_name'],object_id=r['object_id'],attribute_type=r['attribute_type'],label=r['label'],intervention=kind,original_score=original,changed_score=changed,score_change=changed-original,original_preferred=original>changed,label_status=r['label_status'],gate='invalid_black RGB support is zero; excluded as Unknown by P0 independent of semantic model'))
    summary={}
    for kind in ['occluded_center','invalid_black']:
        rs=[r for r in pred if r['intervention']==kind];summary[kind]=dict(original_roi_preference=intervals([r['original_preferred'] for r in rs],[r['map_name'] for r in rs]),mean_score_change=intervals([r['score_change'] for r in rs],[r['map_name'] for r in rs]))
    (out/'pixel_interventions_seen_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in pred));dest.write_text(json.dumps(dict(summary=summary,seconds=time.perf_counter()-start,frozen_sha256=sha(data/'frozen.json'),note='fixed Seen-only pixel perturbations; weak target words; no claim of attribute correctness or absent-object evidence'),indent=2));print(json.dumps(summary))
if __name__=='__main__':main()
