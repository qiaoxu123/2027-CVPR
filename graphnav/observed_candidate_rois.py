"""Environment-only native ROI sampling inside the CURRENT camera footprint.
Model-facing outputs are RGB crops/quality only, never raw map rasters or labels.
"""
import cv2
import numpy as np
from .renderer import CityNavObservedViewRenderer
from .memory import ObservedRGBMemory
from .landmark_projection import world_to_frame


def camera_support_mask(roi_quad, camera_footprint, size):
    dst=np.float32([[0,0],[size-1,0],[size-1,size-1],[0,size-1]])
    matrix=cv2.getPerspectiveTransform(np.asarray(roi_quad,np.float32),dst)
    px=cv2.perspectiveTransform(np.asarray(camera_footprint,np.float32)[None],matrix)[0]
    mask=np.zeros((size,size),np.uint8);cv2.fillConvexPoly(mask,np.rint(px).clip(-1000000,1000000).astype(np.int32),1)
    return mask.astype(bool)


def hide_unobserved(rgb,valid,roi_quad,camera_footprint):
    known=np.asarray(valid,bool)&camera_support_mask(roi_quad,camera_footprint,len(rgb))
    out=np.asarray(rgb).copy();out[~known]=0
    return out,known


class CurrentCandidateROIEnvironment:
    def __init__(self,rgbd,directory,min_support=.9):
        self.renderer=CityNavObservedViewRenderer(rgbd,directory,image_size=384)
        self.min_support=float(min_support)

    def current(self,map_name,pose,ground,centers_world,extents=(20.,40.)):
        """Only actual current pose; no GT, target identity or future pose accepted."""
        r=self.renderer;obs=r.current_rgbd(map_name,np.asarray(pose),ground)
        gsd=max(np.hypot(r._raster.transform.a,r._raster.transform.d),np.hypot(r._raster.transform.b,r._raster.transform.e))
        if r._rgb.shape[:2]!=r._height.shape:raise ValueError('RGB/height grid mismatch')
        centers=np.asarray(centers_world,np.float32);pixels=world_to_frame(centers,obs['footprint_world'],obs['bgr'].shape);records=[]
        for i,(center,pixel) in enumerate(zip(centers,pixels)):
            if not np.isfinite(pixel).all() or not ((pixel>=0)&(pixel<=383)).all():continue
            x,y=np.rint(pixel).astype(int)
            if not obs['depth_valid'][y,x]:continue
            for scale,extent in enumerate(extents):
                quad=ObservedRGBMemory.view_corners(center,float(pose[3]),extent/2)
                src=np.asarray([[r._raster.index(float(x),float(y))[1],r._raster.index(float(x),float(y))[0]] for x,y in quad],np.float32)
                native_size=max(2,min(2048,int(round(extent/gsd))+1));dst=np.float32([[0,0],[native_size-1,0],[native_size-1,native_size-1],[0,native_size-1]]);M=cv2.getPerspectiveTransform(src,dst)
                native=cv2.warpPerspective(r._rgb,M,(native_size,native_size),flags=cv2.INTER_LINEAR);height=cv2.warpPerspective(r._height,M,(native_size,native_size),flags=cv2.INTER_LINEAR);valid=cv2.warpPerspective(r._height_valid,M,(native_size,native_size),flags=cv2.INTER_LINEAR)>=.999;valid&=(height<pose[2])&(native.max(-1)>0);native,nvalid=hide_unobserved(native,valid,quad,obs['footprint_world'])
                px=world_to_frame(quad,obs['footprint_world'],obs['bgr'].shape).astype(np.float32);low_size=max(2,int(round(np.linalg.norm(px[1]-px[0])))+1);ldst=np.float32([[0,0],[low_size-1,0],[low_size-1,low_size-1],[0,low_size-1]]);L=cv2.getPerspectiveTransform(px,ldst);low=cv2.warpPerspective(obs['bgr'],L,(low_size,low_size),flags=cv2.INTER_LINEAR);lvalid=cv2.warpPerspective(obs['depth_valid'].astype(np.float32),L,(low_size,low_size),flags=cv2.INTER_LINEAR)>=.999;low,lvalid=hide_unobserved(low,lvalid,quad,obs['footprint_world']);common=bool(nvalid.mean()>=self.min_support and lvalid.mean()>=self.min_support)
                records.append(dict(candidate=i,scale=scale,extent_m=extent,quad=quad,native=native,low=low,valid=common,native_support=float(nvalid.mean()),low_support=float(lvalid.mean()),native_size=native_size,low_size=low_size,source_gsd_m=float(gsd),native_effective_gsd_m=float(extent/max(native_size-1,1))))
        return obs,records
