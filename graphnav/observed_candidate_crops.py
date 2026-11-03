"""Exact current-image ROI slices. No map, target, renderer or future access."""
import numpy as np


def observed_candidate_crops(rgb,centers,valid,window_sizes=(48,112)):
    rgb=np.asarray(rgb);centers=np.asarray(centers);valid=np.asarray(valid)
    if rgb.shape!=(384,384,3) or rgb.dtype!=np.uint8 or centers.shape!=(len(valid),2) or valid.dtype!=bool:
        raise ValueError('actual uint8 384 RGB, pixel centers and bool validity required')
    if not np.isfinite(centers).all() or any(int(s)!=s or s<=0 for s in window_sizes):raise ValueError('finite centers and positive integer windows required')
    crops=[];metadata=[]
    for index in np.flatnonzero(valid):
        for scale,size in enumerate(window_sizes):
            x,y=np.floor(centers[index]+.5-size/2).astype(int);right=x+size;bottom=y+size
            x0,y0=max(0,x),max(0,y);x1,y1=min(384,right),min(384,bottom)
            if x0>=x1 or y0>=y1:continue
            crops.append(np.ascontiguousarray(rgb[y0:y1,x0:x1]))
            metadata.append(dict(candidate=int(index),scale=scale,window_pixels=int(size),box_xyxy=[int(x0),int(y0),int(x1),int(y1)],clipped=bool(x0!=x or y0!=y or x1!=right or y1!=bottom)))
    return crops,metadata
