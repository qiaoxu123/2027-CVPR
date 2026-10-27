"""Label-free current native384 RGB -> WALDO boxes -> language/geometry pack."""
from collections import OrderedDict
import hashlib
import time
import cv2
import numpy as np
import torch
from PIL import Image
from .paths import WALDO
from .core import box_landmark_geometry
from .mentions import landmark_mention_spans
from .observed_geometry import sample_surface, local_support


class FrozenBoxCropEncoder:
    """Actual box crops using shared frozen SigLIP weights and explicit HWC RGB."""
    def __init__(self,native):
        self.model,self.processor=native.model,native.processor
        self.device,self.dimension=native.device,native.dimension
        self.cache=OrderedDict()
        self.microbatch=16
        self.cache_hits=self.encoded_images=0

    @torch.inference_mode()
    def images(self,views):
        if not views:
            return torch.empty(0,self.dimension,device=self.device)
        keys=[]
        for view in views:
            digest=hashlib.sha256(str(view.shape).encode())
            digest.update(np.ascontiguousarray(view).tobytes())
            keys.append(digest.hexdigest())
        missing={key:view for key,view in zip(keys,views) if key not in self.cache}
        self.cache_hits+=len(keys)-len(missing)
        pairs=list(missing.items())
        for start in range(0,len(pairs),self.microbatch):
            chunk=pairs[start:start+self.microbatch]
            images=[Image.fromarray(cv2.cvtColor(view,cv2.COLOR_BGR2RGB)) for _,view in chunk]
            # HF channel inference treats a height/width of 3 as CHW. Explicit
            # channels_last is essential for real tiny detection crops.
            batch=self.processor(images=images,input_data_format='channels_last',return_tensors='pt').to(self.device)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.device.type=='cuda'):
                result=self.model.get_image_features(**batch)
            if not torch.is_tensor(result):
                result=result.pooler_output
            result=torch.nn.functional.normalize(result.float(),dim=-1)
            for (key,_),vector in zip(chunk,result):
                self.cache[key]=vector.detach().cpu().clone()
                self.encoded_images+=1
        result=torch.stack([self.cache[key] for key in keys]).to(self.device)
        for key in keys:
            self.cache.move_to_end(key)
        while len(self.cache)>4096:
            self.cache.popitem(last=False)
        return result


class ObservedBoxPerception:
    def __init__(self, native):
        from ultralytics import YOLO
        self.detector=YOLO(str(WALDO))
        self.detector.model.eval().requires_grad_(False)
        # Share frozen weights but use an independent crop cache/batch setting;
        # do not change A0's image encoder, cache or microbatch configuration.
        self.crops=FrozenBoxCropEncoder(native)

    @torch.no_grad()
    def extract(self, policy, state, field):
        from graphnav.landmark_projection import frame_to_world, world_to_frame
        tick=time.perf_counter()
        ob=state.current_observation
        image=ob['bgr']
        if image.shape!=(384,384,3):
            raise ValueError('A2 requires actual native384 RGB, no higher-resolution rerender')
        rgb_valid=image.max(-1)>0
        surface=np.asarray(ob['pose'])[2]-ob['depth_m']
        depth_valid=ob['depth_valid'] & rgb_valid
        grid=policy.world[0].float().cpu().numpy()
        base=field.belief_logits[0].float()
        projected=world_to_frame(grid,ob['footprint_world'],image.shape)
        inside=np.isfinite(projected).all(-1) & (projected>=0).all(-1) & (projected<=383).all(-1)
        observed=np.zeros(len(grid),bool)
        pix=np.rint(projected[inside]).astype(int)
        observed[inside]=rgb_valid[pix[:,1],pix[:,0]]
        candidates=[]
        raw_count=0
        detector_error=None
        if rgb_valid.any():
            try:
                predictions=self.detector.predict(image,imgsz=1024,conf=.25,quantize=16,
                                                  device=0,verbose=False,max_det=300)[0]
            except (RuntimeError, ValueError) as error:
                # Record failure explicitly. OOM is fatal: silently swallowing
                # it could leave the process unable to evaluate later episodes.
                if 'out of memory' in str(error).lower():
                    raise
                predictions=None
                detector_error=type(error).__name__+': '+str(error)
            if predictions is not None:
                raw_count=len(predictions.boxes)
                for box in predictions.boxes:
                    name=predictions.names[int(box.cls)]
                    kind=0 if name in ('LightVehicle','Truck','Bus','UtilityVehicle') else 1 if name=='Building' else -1
                    if kind<0:
                        continue
                    x0,y0,x1,y1=np.clip(box.xyxy[0].cpu().numpy(),0,383)
                    lo=np.floor([x0,y0]).astype(int)
                    hi=np.minimum(np.ceil([x1,y1]).astype(int)+1,384)
                    if (hi-lo<3).any():
                        continue
                    crop=image[lo[1]:hi[1],lo[0]:hi[0]]
                    if (crop.max(-1)>0).mean()<.2:
                        continue
                    pixels=np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]],np.float32)
                    polygon=frame_to_world(pixels,ob['footprint_world'],image.shape)
                    back=world_to_frame(polygon,ob['footprint_world'],image.shape)
                    if not np.isfinite(polygon).all() or not np.allclose(back,pixels,atol=.02):
                        raise ValueError('Box projection roundtrip failed')
                    candidates.append(dict(poly=polygon,xy=polygon.mean(0),pixels=pixels,
                                           kind=kind,detector_class=name,confidence=float(box.conf),
                                           size=polygon.max(0)-polygon.min(0),crop=crop.copy(),
                                           surface=sample_surface(surface,depth_valid,pixels)))
        detector_seconds=time.perf_counter()-tick
        visual=self.crops.images([c['crop'] for c in candidates]).float()
        references=policy.a2_references
        offsets=policy.bert.tokenizer(policy.record['instruction'],truncation=True,max_length=policy.bert.max_length,
                                      return_offsets_mapping=True)['offset_mapping']
        if len(offsets)!=policy.tokens.shape[1]:
            raise ValueError('A0 BERT token sequence and A2 mention offsets differ')
        mentions,linked=landmark_mention_spans(policy.record['instruction'],offsets,references)
        chosen=np.flatnonzero(linked)
        refs=[references[j] for j in chosen]
        mention=mentions[chosen]
        polys=[np.asarray(r['contour'],np.float32) for r in refs]
        heights=[sample_surface(surface,depth_valid,world_to_frame(p,ob['footprint_world'],image.shape)) for p in polys]
        if refs:
            names,nmask=policy.bert.encode_tokens([r['name']+'; '+r.get('semantic_description','') for r in refs])
            names,nmask=names.float(),nmask.bool()
        else:
            names=torch.zeros(1,1,768,device=policy.device)
            nmask=torch.zeros(1,1,dtype=torch.bool,device=policy.device)
            mention=np.zeros((1,len(offsets)),bool)
        pair,pair_valid=box_landmark_geometry([c['poly'] for c in candidates],polys,[c['confidence'] for c in candidates],
                                            box_height_m=[c['surface']['z'] for c in candidates],
                                            landmark_height_m=[s['z'] for s in heights])
        n,l=len(candidates),len(refs)
        geometry=np.zeros((max(n,1),max(l,1),14),np.float32)
        geometry[:n,:l,:12]=pair
        # Current footprint scale is public camera geometry, not target data.
        metres_per_pixel=float(np.linalg.norm(np.asarray(ob['footprint_world'])[1]-ob['footprint_world'][0])/383)
        geometry[:n,:l,12]=metres_per_pixel/40
        geometry[:n,:l,13]=np.array([c['surface']['coverage'] for c in candidates])[:,None]
        valid_pair=np.zeros((max(n,1),max(l,1)),bool)
        valid_pair[:n,:l]=pair_valid
        support=np.zeros((max(n,1),len(grid)),bool)
        support[:n]=local_support(grid,[c['poly'] for c in candidates],observed)
        xy=np.zeros((max(n,1),2),np.float32)
        limit=np.zeros_like(xy)
        if n:
            xy[:n]=np.array([c['xy'] for c in candidates])
            limit[:n]=np.minimum(np.array([c['size'] for c in candidates])/2+2,8)
        def tensor(value,dtype=torch.float32):
            return torch.as_tensor(value,device=policy.device,dtype=dtype)
        rgb=torch.zeros(max(n,1),policy.native.dimension,device=policy.device)
        if n:
            rgb[:n]=visual
        network=dict(instruction_tokens=policy.tokens[0].float(),token_valid=policy.mask[0].bool(),
                     mention_mask=tensor(mention,torch.bool),landmark_name_tokens=names,name_valid=nmask,
                     link_valid=tensor(np.ones(l,bool) if l else np.zeros(1,bool),torch.bool),
                     box_rgb=rgb,box_valid=tensor(np.arange(max(n,1))<n,torch.bool),
                     pair_geometry=tensor(geometry),pair_valid=tensor(valid_pair,torch.bool),offset_limit=tensor(limit))
        spatial=dict(base_logits=base,grid_world=tensor(grid),candidate_xy=tensor(xy),
                     candidate_support=tensor(support,torch.bool),observed=tensor(observed,torch.bool))
        def plain(record):
            return {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in record.items() if k!='crop'}
        meta=dict(candidates=[plain(c) for c in candidates],references=refs,
                  reference_surfaces=heights,public_reference_records=len(references),
                  valid_references=l,raw_detection_count=raw_count,detector_cap_reached=raw_count>=300,
                  detector_error=detector_error,detector_seconds=detector_seconds,
                  extraction_seconds=time.perf_counter()-tick,metres_per_pixel=metres_per_pixel,
                  frame_shape=list(image.shape),pose=state.pose.tolist(),
                  footprint_world=np.asarray(ob['footprint_world']).tolist())
        return dict(network=network,spatial=spatial),meta
