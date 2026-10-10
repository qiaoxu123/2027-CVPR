"""RefSegRS mask-supervised fixed-tile pilot, not a claimed segmentation model.

All 64 equal tiles are generated without reading the mask. Template-derived
query masks supervise multi-positive tile presence; ambiguous tiles are Unknown.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile
import cv2
import numpy as np
import torch
from .city_features import CropEncoder
from graphnav.checkpointing import file_sha256


def run(args):
    cv2.setNumThreads(1);root=Path(args.output)
    if root.exists():raise ValueError('refusing mask-feature overwrite')
    root.mkdir(parents=True);started=time.time();encoder=CropEncoder();rows=[];files=[];counts=dict(images=0,usable=0,no_usable_mask=0,regions=0)
    with zipfile.ZipFile(args.archive) as archive:
        annotations=[line.split(' ',1) for line in archive.read('RefSegRS/output_phrase_'+args.split+'.txt').decode().splitlines() if line.strip()]
        annotations=sorted(annotations,key=lambda row:hashlib.sha256(('refsegpilot:'+row[0]).encode()).hexdigest())[:args.limit]
        for index,(identity,phrase) in enumerate(annotations):
            image=archive.read('RefSegRS/images/'+identity+'.tif');maskbytes=archive.read('RefSegRS/masks/'+identity+'.tif')
            bgr=cv2.imdecode(np.frombuffer(image,np.uint8),cv2.IMREAD_COLOR);mask=cv2.imdecode(np.frombuffer(maskbytes,np.uint8),cv2.IMREAD_UNCHANGED)
            if bgr.shape!=(512,512,3) or mask.shape not in [(512,512),(512,512,3)]:raise ValueError('unexpected official RefSegRS dimensions')
            if mask.ndim==3:
                if not np.array_equal(mask[:,:,0],mask[:,:,1]) or not np.array_equal(mask[:,:,0],mask[:,:,2]):raise ValueError('nonbinary mask channels')
                mask=mask[:,:,0]
            if not set(np.unique(mask)).issubset({0,255}):raise ValueError('unsupported nonbinary query mask')
            boxes=[(x,y,x+64,y+64) for y in range(0,512,64) for x in range(0,512,64)]
            coverage=np.array([(mask[y:d,x:c]>0).mean() for x,y,c,d in boxes]);positive=coverage>=.02;negative=coverage<=.001
            counts['images']+=1
            if not positive.any() or not negative.any():counts['no_usable_mask']+=1;continue
            vectors,tokens=encoder.encode([bgr[y:d,x:c].copy() for x,y,c,d in boxes],tokens=True)
            text=encoder.native.texts([phrase]).cpu().half()[0]
            batch=encoder.native.processor(text=[phrase],padding='max_length',truncation=True,max_length=encoder.native.model.config.text_config.max_position_embeddings,return_tensors='pt').to(encoder.device)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):words=encoder.native.model.text_model(**batch).last_hidden_state[0].cpu().half()
            valid=batch['input_ids'][0].cpu()!=encoder.native.processor.tokenizer.pad_token_id
            objects=[dict(object_id=str(i),object_type='mask_tile',bbox=list(box),native_min_side=64) for i,box in enumerate(boxes)]
            rows.append(dict(metadata=dict(sample_id='refseg:'+identity,family_id='refseg:'+hashlib.sha256(image).hexdigest(),map_name='refseg_official_'+args.split,direction='forward',instruction=phrase,roi_objects=objects),public=dict(roi_valid=torch.ones(64,dtype=torch.bool),roi_quality=torch.tensor([[.125,1,1]]*64)),labels=dict(positive_roi_mask=torch.from_numpy(positive),semantic_valid_mask=torch.from_numpy(positive|negative)),features=dict(tight=vectors,polygon_masked=vectors),patch_tokens=tokens,texts=dict(full=text,phrase=text),language_tokens=words,language_valid=valid,phrase=phrase,phrase_supported=True,mask_coverage=coverage.tolist(),image_sha256=hashlib.sha256(image).hexdigest(),annotation_status='template expression + derived semantic query mask, fixed tiles; not human CityNav visual attributes'))
            counts['usable']+=1;counts['regions']+=64
            if len(rows)==16:
                path=root/f'chunk_{len(files):04}.pt';torch.save(rows,path);files.append(path.name);rows=[]
            if index%25==0:print(json.dumps(dict(counts,seconds=time.time()-started)),flush=True)
        if rows:
            path=root/f'chunk_{len(files):04}.pt';torch.save(rows,path);files.append(path.name)
    manifest=dict(complete=True,chunks=files,files={p:file_sha256(root/p) for p in files},counts=counts,seconds=time.time()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated(),archive_sha256=file_sha256(args.archive),split=args.split,annotation_policy='official masks: tile coverage>=2% positive, <=0.1% negative, middle Unknown; all 64 fixed tiles made without labels; mask used only supervision',official_scene_split='published scene-disjoint official Train/Val; no GPS or scene map supplied in archive, image SHA identities checked by trainer',mask_policy='uniform RGB tiles, no target-conditioned crop or mask input')
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--split',choices=['train','val'],required=True);p.add_argument('--limit',type=int,default=192);p.add_argument('--output',required=True);run(p.parse_args())


if __name__=='__main__':main()
