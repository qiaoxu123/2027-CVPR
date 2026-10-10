"""Uniform representations of actual observed ROIs; labels never crop images."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from experiments.bidirectional_core.cache import CachedCoreDataset,DEFAULT_DATA
from graphnav.appearance_matching import ObservedSigLIPMatchingEncoder
from graphnav.renderer import CityNavObservedViewRenderer
from graphnav.checkpointing import file_sha256
from experiments.visual_grounding.data.audit_attributes import bind_phrase


def visual_phrase(instruction):
    """First named target noun only; ambiguous attributes stay full language."""
    import re
    nouns=list(re.finditer(r'\b(building|car|vehicle|truck|van)\b',instruction.lower()))
    if not nouns:return instruction,False
    target=nouns[0];typ='Building' if target.group()=='building' else 'Car'
    # Subordinate nearby-object clauses must not furnish target colors.
    clause=re.split(r'\b(?:beside|behind|next to|in front of|with|near)\b',instruction,flags=re.I)[0]
    attrs=bind_phrase(clause,typ)
    if not attrs:return instruction,False
    order=['roof_color','building_color','vehicle_color','building_size','building_shape']
    words=[]
    for key in order:
        if key in attrs:words.append(str(attrs[key])+(' roof' if key=='roof_color' else ''))
    if not words:return instruction,False
    return 'a '+' '.join(words)+' '+target.group(),True


class CropEncoder:
    def __init__(self,device='cuda'):
        self.native=ObservedSigLIPMatchingEncoder(device=device,microbatch=16)
        self.device=torch.device(device)

    @torch.inference_mode()
    def encode(self,views,tokens=False):
        vectors=[];patches=[]
        for off in range(0,len(views),16):
            batch=self.native.processor(images=[cv2.cvtColor(v,cv2.COLOR_BGR2RGB) for v in views[off:off+16]],return_tensors='pt').to(self.device)
            with torch.autocast('cuda',dtype=torch.float16):
                hidden=self.native.model.vision_model(**batch).last_hidden_state
            with torch.autocast('cuda',enabled=False):
                pooled=self.native.model.vision_model.head(hidden.float())
            vectors.append(F.normalize(pooled.float(),dim=-1).cpu().half())
            if tokens:
                local=hidden.transpose(1,2).reshape(-1,1024,24,24).float()
                patches.append(F.adaptive_avg_pool2d(local,(4,4)).flatten(2).transpose(1,2).cpu().half())
        return torch.cat(vectors),torch.cat(patches) if tokens else None


def representations(bgr,region,polygon,footprint):
    from graphnav.landmark_projection import project_landmark
    x0,y0,x1,y1=map(int,region['bbox']);tight=bgr[y0:y1,x0:x1].copy()
    if not tight.size:raise ValueError('empty actual ROI')
    w,h=x1-x0,y1-y0
    a=max(0,int(x0-.25*w));b=max(0,int(y0-.25*h));c=min(384,int(x1+.25*w));d=min(384,int(y1+.25*h))
    context=bgr[b:d,a:c].copy()
    projected=project_landmark(polygon,footprint,bgr.shape,min_visible=.9,min_pixels=256)
    if projected is None or list(projected.bbox_xyxy)!=list(region['bbox']):
        raise ValueError('cached crop does not match actual projected polygon')
    mask=projected.mask[y0:y1,x0:x1].astype(bool)
    masked=tight.copy();masked[~mask]=127
    return tight,masked,context,float(mask.mean())


def run(args):
    cv2.setNumThreads(1);started=time.time();root=Path(args.output)
    if root.exists():raise ValueError('refusing feature overwrite')
    root.mkdir(parents=True)
    data=CachedCoreDataset(args.cache,'E1');selected=[]
    for i in range(len(data)):
        row=data[i];p=row['labels']['positive_roi_mask'];v=row['labels']['semantic_valid_mask']
        if args.selection=='judged':
            eligible=bool(p.any() and (v & ~p).any())
        else:
            _,supported=visual_phrase(row['metadata']['instruction'])
            # Public inference applicability must never depend on true target
            # visibility or weak labels. Include every supported query/view.
            eligible=bool(supported and row['public']['roi_valid'].any())
        if eligible:selected.append(row)
    if args.limit:selected=selected[:args.limit]
    raw=json.loads((Path(args.data_root)/'cityrefer/objects.json').read_text())
    encoder=CropEncoder();renderer=CityNavObservedViewRenderer(Path(args.data_root)/'rgbd',root/'renderer_private',image_size=384)
    chunks=[];written=[];roi_count=0
    for index,row in enumerate(selected):
        public=row['public'];meta=row['metadata'];pose=np.asarray(meta['actual_camera_pose'],np.float64)
        observation=renderer.current_rgbd(meta['map_name'],pose,float(public['ground_level']));regions=meta['roi_objects']
        crops=[representations(observation['bgr'],region,dict(id=region['object_id'],object_type=region['object_type'],name=region.get('name',''),contour=raw[meta['map_name']][region['object_id']]['contour']),observation['footprint_world']) for region in regions]
        n=len(crops)
        vectors,tokens=encoder.encode([c[0] for c in crops],tokens=True)
        masked,_=encoder.encode([c[1] for c in crops]);context,_=encoder.encode([c[2] for c in crops]);whole,_=encoder.encode([observation['bgr']])
        phrase,supported=visual_phrase(meta['instruction'])
        text=encoder.native.texts([meta['instruction'],phrase]).float().cpu().half()
        language_batch=encoder.native.processor(text=[meta['instruction']],padding='max_length',truncation=True,max_length=encoder.native.model.config.text_config.max_position_embeddings,return_tensors='pt').to(encoder.device)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            language_tokens=encoder.native.model.text_model(**language_batch).last_hidden_state[0].cpu().half()
        language_valid=language_batch['input_ids'][0].cpu()!=encoder.native.processor.tokenizer.pad_token_id
        result=dict(metadata=meta,labels=row['labels'],public=public,features=dict(tight=vectors,polygon_masked=masked,context=context,multiscale=F.normalize(vectors.float()+context.float(),dim=-1).half(),whole=whole.expand(n,-1).clone(),patch_pool=public['roi_features']),patch_tokens=tokens,
            texts=dict(full=text[0],phrase=text[1]),language_tokens=language_tokens,language_valid=language_valid,
            phrase=phrase,phrase_supported=supported,polygon_bbox_occupancy=[c[3] for c in crops])
        if not all(torch.isfinite(t).all() for t in [vectors,masked,context,whole,tokens,text]):raise ValueError('nonfinite CUDA output')
        chunks.append(result);roi_count+=n
        if len(chunks)==16 or index==len(selected)-1:
            path=root/f'chunk_{len(written):04}.pt';torch.save(chunks,path);written.append(path.name);chunks=[]
        if index%25==0:print(json.dumps(dict(states=index+1,total=len(selected),rois=roi_count,seconds=time.time()-started)),flush=True)
    manifest=dict(complete=True,states=len(selected),rois=roi_count,chunks=written,files={p:file_sha256(root/p) for p in written},source_manifest_sha256=file_sha256(Path(args.cache)/'manifest.json'),seconds=time.time()-started,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),precision='fp16 image transformer, fp32 native pooling',siglip_revision=getattr(encoder.native.model.config,'_commit_hash',None),
        selection=args.selection,
        protocol='same current pose and candidate ROI identity order, source384 only; no detail from original full raster; masked polygon is geometric weak mask; text phrase binds from supplied instruction only')
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--output',required=True);p.add_argument('--data-root',default=DEFAULT_DATA);p.add_argument('--limit',type=int);p.add_argument('--selection',choices=['judged','public-visual'],default='judged');run(p.parse_args())


if __name__=='__main__':main()
