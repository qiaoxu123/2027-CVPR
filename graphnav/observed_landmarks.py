"""Current-observation ROI evidence, independent of destination and Top-K goals.

The polygon is a weak registration proposal. Quality below is source coverage
and boundary sensitivity, never a semantic segmentation accuracy estimate.
"""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
from pathlib import Path
import cv2
import numpy as np
import torch
from .landmark_projection import project_landmark,make_roi_image


@dataclass
class ObservedLandmark:
    object_id: str
    name: str
    object_type: str
    contour_world: np.ndarray
    roi_image: np.ndarray
    confidence: float
    quality: dict


def current_landmark_rois(observation,landmarks,*,max_rois=32):
    bgr=np.asarray(observation['bgr']);depth_valid=np.asarray(observation['depth_valid'],dtype=bool)
    if bgr.shape[:2]!=(384,384):raise ValueError('native 384px current observation required')
    if depth_valid.shape!=bgr.shape[:2]:raise ValueError('depth/RGB observation dimensions disagree')
    support=bgr.max(-1)>0;results=[]
    for landmark in landmarks:
        roi=project_landmark(landmark,observation['footprint_world'],bgr.shape,min_visible=.9,min_pixels=256)
        if roi is None:continue
        x0,y0,x1,y1=roi.bbox_xyxy
        if min(x1-x0,y1-y0)<8:continue
        mask=roi.mask.astype(bool);rgb_support=float(support[mask].mean());depth_support=float(depth_valid[mask].mean())
        if rgb_support<.9:continue
        eroded=cv2.erode(roi.mask,np.ones((5,5),np.uint8))
        stable_fraction=float(eroded.sum()/max(roi.visible_pixels,1))
        # Missing depth reduces confidence; it is not evidence that an object is absent.
        confidence=roi.visible_fraction*rgb_support*stable_fraction*(.5+.5*depth_support)
        if confidence<=0:continue
        quality=dict(projected_visible_fraction=roi.visible_fraction,rgb_source_support=rgb_support,
            vertical_depth_support=depth_support,two_pixel_stable_fraction=stable_fraction,
            visible_pixels=roi.visible_pixels,semantic_agreement='unverified geometric proposal')
        results.append(ObservedLandmark(roi.object_id,roi.name,roi.object_type,
            np.asarray(landmark['contour'],dtype=np.float32),make_roi_image(bgr,roi,output_size=384),confidence,quality))
    # Quality cap is unrelated to instruction, candidate rank, target visibility or labels.
    results.sort(key=lambda item:(-item.confidence*item.quality['visible_pixels'],item.object_id))
    return results[:max_rois]


class ObservedLandmarkEncoder:
    """Cache only features of actual current pixels; no offline/future gallery."""
    def __init__(self,checkpoint,directory,*,device='cuda',cache_size=8192):
        from .landmark_visual_pretrain import FrozenSigLIP2Patches,VisualLandmarkAligner
        state=torch.load(checkpoint,map_location='cpu',weights_only=True);args=state['args']
        self.device=torch.device(device);self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.backbone=FrozenSigLIP2Patches(args['image_backbone'],device=device,cache_limit=cache_size,
            microbatch=args['image_microbatch'],cache_directory=self.directory/'patches')
        self.head=VisualLandmarkAligner(vision_dim=self.backbone.hidden_size,hidden=args['hidden'],heads=args['heads']).to(self.device).eval()
        self.head.load_state_dict(state['model'],strict=True);self.head.requires_grad_(False)
        self.vectors=OrderedDict();self.cache_size=cache_size;self.last_step=-1;self.history=[]

    def start_episode(self):
        self.last_step=-1;self.history=[] # Feature cache is pixel-only; evidence history is episode-local.

    @torch.no_grad()
    def observe(self,observation,landmarks,*,step):
        if step<=self.last_step:raise ValueError('actual observations must have strictly increasing steps')
        rois=current_landmark_rois(observation,landmarks);keys=[];paths=[]
        for roi in rois:
            key=hashlib.sha256(roi.roi_image.tobytes()).hexdigest();keys.append(key)
            path=self.directory/(key+'.png')
            if not path.exists() and not cv2.imwrite(str(path),roi.roi_image):raise IOError('current ROI save failed')
            if key not in self.vectors:paths.append((key,str(path)))
        paths=list(dict.fromkeys(paths))
        if paths:
            patches,mask=self.backbone.encode([path for _,path in paths])
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.device.type=='cuda'):
                vectors=self.head(patches,mask).float().cpu()
            if not torch.isfinite(vectors).all():raise ValueError('nonfinite observed landmark vectors')
            for (key,_),vector in zip(paths,vectors):self.vectors[key]=vector
        vectors=torch.stack([self.vectors[key] for key in keys]) if keys else torch.zeros(0,self.head.query.shape[-1])
        for key in keys:self.vectors.move_to_end(key)
        while len(self.vectors)>self.cache_size:self.vectors.popitem(last=False)
        self.last_step=step
        self.history.append(dict(step=step,pose=np.asarray(observation['pose']).tolist(),pixel_keys=keys,
            landmarks=[dict(name=roi.name,object_type=roi.object_type,confidence=roi.confidence,quality=roi.quality) for roi in rois]))
        return rois,vectors
