"""Exact project_landmark arithmetic for catalog polygons already validated once."""
import cv2
import numpy as np
from graphnav.landmark_projection import LandmarkROI,world_to_frame

def project_validated(landmark,footprint_world,image_shape,min_visible=.9,min_pixels=256):
    try:px=world_to_frame(landmark['contour'],footprint_world,image_shape)
    except (ValueError,cv2.error):return None
    if not np.isfinite(px).all() or np.max(np.abs(px))>1e6:return None
    area=abs(float(cv2.contourArea(px)))
    if area<min_pixels:return None
    h,w=image_shape[:2];mask=np.zeros((h,w),np.uint8);cv2.fillPoly(mask,[np.rint(px).astype(np.int32)],1);visible=int(mask.sum());fraction=min(1.,visible/max(area,1e-6))
    if fraction<min_visible or visible<min_pixels:return None
    x,y,bw,bh=cv2.boundingRect(mask)
    if bw==0 or bh==0:return None
    return LandmarkROI(str(landmark['id']),str(landmark.get('name','')),str(landmark.get('object_type','Unknown')),px,mask,fraction,area,visible,(x,y,x+bw,y+bh))
