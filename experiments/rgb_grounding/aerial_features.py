"""Encode official AerialVG annotations without inventing instance negatives.

Exact region phrases can refer to multiple boxes. Known negatives require a
contrary explicit color on the same conservative noun class; otherwise Unknown.
Full-caption all-mentioned-object records are audit-only, not primary identity.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import time
import cv2
import numpy as np
import torch
from .city_features import CropEncoder
from graphnav.checkpointing import file_sha256

COLORS=r'white|black|gray|grey|red|blue|yellow|green|brown|silver|orange'


def attributes(phrase):
    text=phrase.lower().replace('grey','gray');colors=set(re.findall(r'\b('+COLORS+r')\b',text));cls=None
    for noun in ['sedan','suv','van','truck','bus','pedestrian','bicycle','tricycle','car']:
        if re.search(r'\b'+noun+r'\b',text):cls=noun;break
    if cls in ['sedan','suv','car']:cls='car'
    return cls,next(iter(colors)) if len(colors)==1 else None


def build(args):
    cv2.setNumThreads(1);output=Path(args.output)
    if output.exists():raise ValueError('refusing external feature overwrite')
    output.mkdir(parents=True);started=time.time();encoder=CropEncoder();by_image=defaultdict(list)
    for r in map(json.loads,Path(args.annotations).read_text().splitlines()):by_image[r['filename']].append(r)
    chunks=[];files=[];counts=dict(images=0,regions=0,phrase_queries=0,usable_queries=0,missing_images=0)
    for filename,expressions in sorted(by_image.items()):
        path=Path(args.images)/filename
        if not path.is_file():counts['missing_images']+=1;continue
        bgr=cv2.imread(str(path));h,w=bgr.shape[:2]
        expected=expressions[0]
        if (h,w)!=(expected['height'],expected['width']):raise ValueError('external source image dimensions mismatch')
        regions={};phrase_boxes=defaultdict(set);captions=[]
        for expression in expressions:
            mention=[]
            for region in expression['grounding']['regions']:
                a,b,c,d=region['bbox'];box=(max(0,int(a)),max(0,int(b)),min(w,int(np.ceil(c))),min(h,int(np.ceil(d))))
                if min(box[2]-box[0],box[3]-box[1])<8:continue
                key=','.join(map(str,box));regions[key]=box;phrase=' '.join(region['phrase'].lower().split());phrase_boxes[phrase].add(key);mention.append(key)
            captions.append(dict(language=expression['grounding']['caption'],mentioned_regions=mention))
        keys=sorted(regions)
        if len(keys)<2:continue
        crops=[bgr[regions[key][1]:regions[key][3],regions[key][0]:regions[key][2]].copy() for key in keys]
        vectors,tokens=encoder.encode(crops,tokens=True);counts['images']+=1;counts['regions']+=len(keys)
        region_properties=defaultdict(set)
        for phrase,objects in phrase_boxes.items():
            cls,color=attributes(phrase)
            if cls and color:
                for key in objects:region_properties[key].add((cls,color))
        for phrase,positives in sorted(phrase_boxes.items()):
            cls,color=attributes(phrase);known=set(positives)
            if cls and color:
                for key,attrs in region_properties.items():
                    # Only unanimous explicit contrary appearance is a negative.
                    if key not in positives and attrs and all(c==cls and v!=color for c,v in attrs):known.add(key)
            counts['phrase_queries']+=1
            if not known-positives:continue
            counts['usable_queries']+=1
            text=encoder.native.texts([phrase]).cpu().half()[0]
            batch=encoder.native.processor(text=[phrase],padding='max_length',truncation=True,max_length=encoder.native.model.config.text_config.max_position_embeddings,return_tensors='pt').to(encoder.device)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):words=encoder.native.model.text_model(**batch).last_hidden_state[0].cpu().half()
            word_valid=batch['input_ids'][0].cpu()!=encoder.native.processor.tokenizer.pad_token_id
            roi_objects=[dict(object_id=key,object_type=(next(iter(region_properties[key]))[0] if len(region_properties[key])==1 else 'Unknown'),bbox=list(regions[key]),native_min_side=min(regions[key][2]-regions[key][0],regions[key][3]-regions[key][1])) for key in keys]
            public=dict(roi_valid=torch.ones(len(keys),dtype=torch.bool),roi_quality=torch.tensor([[min(r['native_min_side']/384,1),1,1] for r in roi_objects]))
            labels=dict(positive_roi_mask=torch.tensor([key in positives for key in keys]),semantic_valid_mask=torch.tensor([key in known for key in keys]))
            chunks.append(dict(metadata=dict(sample_id=filename+':'+hashlib.sha256(phrase.encode()).hexdigest()[:12],family_id=filename,map_name=filename.split('_')[1],direction='forward',instruction=phrase,roi_objects=roi_objects),labels=labels,public=public,features=dict(tight=vectors,polygon_masked=vectors),patch_tokens=tokens,texts=dict(full=text,phrase=text),language_tokens=words,language_valid=word_valid,phrase=phrase,phrase_supported=True,
                               annotation_status='official phrase-box association; explicit color/noun-constrained conservative negatives; unknown not negative',image_sha256=file_sha256(path)))
            if len(chunks)>=32:
                target=output/f'chunk_{len(files):04}.pt';torch.save(chunks,target);files.append(target.name);chunks=[]
        if counts['images']%25==0:print(json.dumps(dict(counts,seconds=time.time()-started)),flush=True)
    if chunks:
        target=output/f'chunk_{len(files):04}.pt';torch.save(chunks,target);files.append(target.name)
    manifest=dict(complete=True,chunks=files,files={name:file_sha256(output/name) for name in files},counts=counts,seconds=time.time()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated(),annotations_sha256=file_sha256(args.annotations),
        mask_policy='AerialVG supplies boxes, not pixel masks; external polygon_masked slot is explicit all-rectangle bbox crop, no fake segmentation',
        split='source-video/image disjoint selected within official Train; no official overlapping Val used as independent Dev')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--annotations',required=True);p.add_argument('--images',required=True);p.add_argument('--output',required=True);build(p.parse_args())


if __name__=='__main__':main()
