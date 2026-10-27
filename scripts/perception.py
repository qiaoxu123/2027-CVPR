"""Current-view feature extraction, no target IDs, labels or source raster access."""
from common import configuration,WALDO
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from graphnav.landmark_projection import frame_to_world,world_to_frame
from geometry import centroid,sample_surface,depth_patch,relation_features,triad_features,choose_candidates,local_peaks,grid_box_support
from schema import validate_public


class BoxPerception:
    def __init__(self):self.detector=YOLO(str(WALDO/'WALDO30_yolov8l-p2_1024x1024.pt'))

    def extract(self,policy,state,field):
        c=configuration();ob=getattr(state,'detail_observation',state.current_observation)
        image=ob['bgr'];h,w=image.shape[:2];rgbvalid=image.max(-1)>0
        valid=ob['depth_valid']&rgbvalid;surface=ob['pose'][2]-ob['depth_m']
        origin=float(np.quantile(surface[valid],.1)) if valid.any() else 0.
        device=policy.device;grid=policy.world[0].float().cpu().numpy();base=field.belief_logits[0].float().cpu().numpy()
        peaks,local=local_peaks(grid,base,c['max_peaks'],c['radius'])
        gp=world_to_frame(grid,ob['footprint_world'],image.shape);inside=np.isfinite(gp).all(-1)&(gp>=0).all(-1)&(gp[:,0]<=w-1)&(gp[:,1]<=h-1)
        observed=np.zeros(len(grid),bool);ix=np.rint(gp[inside]).astype(int);observed[inside]=rgbvalid[ix[:,1],ix[:,0]]
        bounds=np.asarray(policy.record['world_bounds']);refs=policy.record['landmarks'][:c['max_references']]
        rp=[np.asarray(r['contour'],np.float32)*(bounds[2:]-bounds[:2])+bounds[:2] for r in refs]
        def record(poly,kind,confidence,xy=None):
            poly=np.asarray(poly,np.float32);pixels=world_to_frame(poly,ob['footprint_world'],image.shape)
            stat=sample_surface(surface,valid,pixels)
            lo=np.maximum(np.floor(pixels.min(0)).astype(int),0);hi=np.minimum(np.ceil(pixels.max(0)).astype(int)+1,[w,h])
            crop=None
            if (hi-lo>=3).all():
                patch=image[lo[1]:hi[1],lo[0]:hi[0]]
                if (patch.max(-1)>0).mean()>=.2:crop=patch.copy()
            xy=centroid(poly) if xy is None else np.asarray(xy,np.float32)
            index=int(np.linalg.norm(grid-xy,axis=-1).argmin())
            return dict(poly=poly,pixels=pixels,xy=xy,kind=kind,confidence=float(confidence),surface=stat,
                area=float(abs(cv2.contourArea(poly))),size=poly.max(0)-poly.min(0),crop=crop,base=float(base[index]),
                depth=depth_patch(surface,valid,pixels,origin))
        references=[record(poly,-1,1.) for poly in rp]
        raw=[]
        if rgbvalid.any():
            pred=self.detector.predict(image,imgsz=1024,conf=.25,quantize=16,device=0,verbose=False)[0]
            for box in pred.boxes:
                name=pred.names[int(box.cls)];kind=0 if name in ['LightVehicle','Truck','Bus','UtilityVehicle'] else (1 if name=='Building' else -1)
                if kind<0:continue
                x0,y0,x1,y1=box.xyxy[0].cpu().numpy();px=np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]],np.float32)
                world=frame_to_world(px,ob['footprint_world'],image.shape);rec=record(world,kind,float(box.conf),world.mean(0))
                if rec['crop'] is not None:raw.append(rec)
        eligible=[i for i,v in enumerate(raw) if np.linalg.norm(peaks-v['xy'],axis=-1).min()<=c['radius']]
        local_records=[raw[i] for i in eligible]
        # Encode full vectors before choosing semantic/diversity candidates.
        visual=policy.native.images([r['crop'] for r in local_records]).float()
        similarities=(visual*policy.appearance_text[0]).sum(-1).cpu().numpy() if len(local_records) else np.empty(0)
        ids=choose_candidates(np.asarray([v['xy'] for v in local_records]).reshape(-1,2),[v['base'] for v in local_records],similarities,c['max_candidates'])
        candidates=[local_records[i] for i in ids];crgb=visual[ids] if ids else visual
        n,l=len(candidates),len(references);dim=policy.native.dimension
        rrgb=torch.zeros(l,dim,device=device);visible=[i for i,r in enumerate(references) if r['crop'] is not None]
        if visible:rrgb[visible]=policy.native.images([references[i]['crop'] for i in visible]).float()
        if l:names,nmask=policy.bert.encode_tokens([r['name'] for r in refs]);names=names[:,:32];nmask=nmask[:,:32]
        else:names=torch.empty(0,0,768,device=device);nmask=torch.empty(0,0,dtype=torch.bool,device=device)
        def array(records,key,shape):return np.asarray([v[key] for v in records],np.float32).reshape((len(records),*shape))
        cxy=array(candidates,'xy',(2,));rxy=array(references,'xy',(2,))
        def geometry(records):
            rows=[]
            for r in records:
                stat=r['surface'];xy=(r['xy']-np.asarray(state.pose[:2]))/c['radius']
                rows.append([*xy,np.log1p(r['size'][0])/5,np.log1p(r['size'][1])/5,np.log1p(r['area'])/10,
                    float(r['kind']==0),float(r['kind']==1),r['confidence'],(r['base']-base.max())/10,
                    float(r['crop'] is not None),float(stat['valid']),stat['coverage']])
            return np.asarray(rows,np.float32).reshape(-1,12)
        def zs(records):return [r['surface']['z'] for r in records]
        def known(records):return [r['surface']['valid'] for r in records]
        image_c=world_to_frame(cxy,ob['footprint_world'],image.shape) if n else np.empty((0,2))
        image_r=world_to_frame(rxy,ob['footprint_world'],image.shape) if l else np.empty((0,2))
        ip=image_c[:,None]-image_r[None];ip=ip/np.maximum(np.linalg.norm(ip,axis=-1,keepdims=True),1e-6)
        pair=relation_features(cxy,rxy,zs(candidates),zs(references),known(candidates),known(references),
            rpolys=rp,carea=[r['area'] for r in candidates],rarea=[r['area'] for r in references],image_direction=ip,radius=c['radius'])
        triad,triad_ids=triad_features(cxy,rxy)
        support=grid_box_support(grid,[v['poly'] for v in candidates],observed,local)
        def tensor(a,dtype=torch.float32):return torch.as_tensor(a,device=device,dtype=dtype)
        network=dict(tokens=policy.tokens[0].float(),token_mask=policy.mask[0].bool(),candidate_rgb=crgb,reference_rgb=rrgb,
            names=names.float(),name_mask=nmask.bool(),candidate_depth=tensor(array(candidates,'depth',(2,32,32))),
            reference_depth=tensor(array(references,'depth',(2,32,32))),candidate_geometry=tensor(geometry(candidates)),reference_geometry=tensor(geometry(references)),
            pair=tensor(pair),triad=tensor(triad),triad_ids=tensor(triad_ids,torch.long),offset_limit=tensor(np.minimum(array(candidates,'size',(2,))/2+2,8)))
        public=dict(network=network,spatial=dict(grid_world=tensor(grid),base_logits=tensor(base),candidate_xy=tensor(cxy),candidate_support=tensor(support,torch.bool),
            local_mask=tensor(local,torch.bool),candidate_base=tensor([v['base'] for v in candidates])))
        def small(r):return {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in r.items() if k not in ['crop','depth']}
        meta=dict(candidates=[small(r) for r in candidates],references=[dict(small(r),name=ref['name']) for r,ref in zip(references,refs)],
            raw_detections=[small(r) for r in raw],local_preselection_ids=eligible,selected_local_ids=ids,peaks=peaks.tolist(),origin_surface_z=origin,
            pose=state.pose.tolist(),frame_shape=list(image.shape),footprint_world=np.asarray(ob['footprint_world']).tolist())
        return validate_public(public),meta
