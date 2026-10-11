"""AerialVG full-caption primary identity bound to OFFICIAL eval.py boxes[0].

The released eval reader, not a guessed ordering, defines region0 as target;
remaining annotated regions are context. Overlapping proposals of the primary
and identical captions remain multi-positive. No arbitrary primary creation.
"""
import argparse,hashlib,json,time
from collections import defaultdict
from pathlib import Path
import cv2,numpy as np,torch
from experiments.rgb_grounding.city_features import CropEncoder
from experiments.rgb_grounding.aerial_features import attributes
from experiments.rgb_grounding.evaluate_features import load
from graphnav.checkpointing import file_sha256
from .data import write

def overlap(a,b):
    c=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));return c/max((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-c,1)

def run(a):
    root=Path(a.output)
    if root.exists():raise ValueError('no overwrite')
    root.mkdir(parents=True);start=time.time();cv2.setNumThreads(1);encoder=CropEncoder();reused={}
    for r in load(a.reuse):
        for j,v in enumerate(r['metadata']['roi_objects']):reused[(r['metadata']['family_id'],v['object_id'])]=dict(image=r['features']['tight'][j],patch=r['patch_tokens'][j])
    byimage=defaultdict(list)
    for r in map(json.loads,Path(a.annotations).read_text().splitlines()):byimage[r['filename']].append(r)
    rows=[];files=[];images=0;rois=0;hits=0
    for filename,expressions in sorted(byimage.items()):
        path=Path(a.images)/filename
        if not path.is_file():continue
        bgr=cv2.imread(str(path));h,w=bgr.shape[:2];boxes={};props={}
        def box(reg):
            v=reg['bbox'];return [max(0,int(v[0])),max(0,int(v[1])),min(w,int(np.ceil(v[2]))),min(h,int(np.ceil(v[3])))]
        for ex in expressions:
            for reg in ex['grounding']['regions']:
                v=box(reg)
                if min(v[2]-v[0],v[3]-v[1])<8:continue
                key=','.join(map(str,v));boxes[key]=v;props[key]=attributes(reg['phrase'])[0] or 'Unknown'
        keys=sorted(boxes)
        if len(keys)<2:continue
        images+=1;rois+=len(keys);tight=[];patch=[];contexts=[]
        for key in keys:
            x,y,c,d=boxes[key];old=reused.get((filename,key))
            if old:tight.append(old['image']);patch.append(old['patch']);hits+=1
            else:
                v,t=encoder.encode([bgr[y:d,x:c]],tokens=True);tight.append(v[0]);patch.append(t[0])
            cw,ch=c-x,d-y;contexts.append(bgr[max(0,int(y-.25*ch)):min(h,int(d+.25*ch)),max(0,int(x-.25*cw)):min(w,int(c+.25*cw))])
        ctx,_=encoder.encode(contexts);tight=torch.stack(tight);patch=torch.stack(patch)
        for ex in expressions:
            full=ex['grounding']['caption'];primary=box(ex['grounding']['regions'][0]);aliases=[box(q['grounding']['regions'][0]) for q in expressions if ' '.join(q['grounding']['caption'].lower().split())==' '.join(full.lower().split())]
            pos=torch.tensor([any(overlap(boxes[k],v)>.5 for v in aliases) for k in keys]);
            if not pos.any() or pos.all():continue
            phrase=ex['grounding']['regions'][0]['phrase'];vec=encoder.native.texts([full,phrase]).cpu().half();batch=encoder.native.processor(text=[full],padding='max_length',truncation=True,max_length=64,return_tensors='pt').to(encoder.device)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):words=encoder.native.model.text_model(**batch).last_hidden_state[0].cpu().half()
            regs=[dict(object_id=k,object_type=props[k],bbox=boxes[k],native_min_side=min(boxes[k][2]-boxes[k][0],boxes[k][3]-boxes[k][1])) for k in keys]
            rows.append(dict(metadata=dict(sample_id=filename+':primary:'+hashlib.sha256(full.encode()).hexdigest()[:16],family_id=filename,map_name=filename.split('_')[1],instruction=full,direction='forward',roi_objects=regs),public=dict(roi_valid=torch.ones(len(keys),dtype=torch.bool),roi_quality=torch.tensor([[min(r['native_min_side']/384,1),1,1] for r in regs])),labels=dict(positive_roi_mask=pos,semantic_valid_mask=torch.ones_like(pos),identity_positive=pos.clone()),features=dict(tight=tight,context=ctx),patch_tokens=patch,texts=dict(full=vec[0],phrase=vec[1]),language_tokens=words,language_valid=batch['input_ids'][0].cpu()!=encoder.native.processor.tokenizer.pad_token_id,primary_annotation='official eval.py gt_box=targets.boxes[0]; same-caption and bboxIoU>.5 proposal aliases multi-positive',image_sha256=file_sha256(path)))
        if len(rows)>=32:
            dest=root/f'chunk_{len(files):04}.pt';torch.save(rows,dest);files.append(dest.name);rows=[]
        if images%25==0:print(json.dumps(dict(images=images,regions=rois,reused=hits,seconds=time.time()-start)),flush=True)
    if rows:
        dest=root/f'chunk_{len(files):04}.pt';torch.save(rows,dest);files.append(dest.name)
    write(root/'manifest.json',dict(complete=True,chunks=files,files={n:file_sha256(root/n) for n in files},images=images,regions=rois,reused_vectors=hits,seconds=time.time()-start,annotation_sha256=file_sha256(a.annotations),official_eval_sha256=file_sha256(a.official_eval),peak_gpu_bytes=torch.cuda.max_memory_allocated(),candidate_policy='all annotated observed bbox proposals; primary boxes[0] verified official evaluation contract, no target ID input; not end-to-end detection recall',unknown_policy='full caption original primary vs supplied observed annotated proposals; aliases kept multi-positive; official referring identity benchmark labels, not independently reverified CityNav attributes'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--annotations',required=True);p.add_argument('--images',required=True);p.add_argument('--reuse',required=True);p.add_argument('--official-eval',required=True);p.add_argument('--output',required=True);run(p.parse_args())
if __name__=='__main__':main()
