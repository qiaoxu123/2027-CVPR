"""Fine spatial hypotheses from actual current 384px observations only."""
import numpy as np
from graphnav.landmark_projection import frame_to_world


def local_patch_regions(observation):
    rgb=np.asarray(observation['bgr']); depth=np.asarray(observation['depth_valid'],bool)
    if rgb.shape!=(384,384,3) or depth.shape!=(384,384):
        raise ValueError('actual native 384px RGB/depth validity required')
    support=(rgb.max(-1)>0)&depth
    patch=support.reshape(24,16,24,16).mean((1,3))>=.9
    gy,gx=np.meshgrid(np.arange(1,24,2),np.arange(1,24,2),indexing='ij')
    centers=np.stack((gx.ravel(),gy.ravel()),-1)
    pixels=(centers+.5)*16-.5
    world=frame_to_world(pixels,observation['footprint_world'],rgb.shape)
    masks=np.zeros((len(centers),3,576),bool)
    valid=patch[centers[:,1],centers[:,0]]
    quality=np.zeros((len(centers),6),np.float32)
    footprint=np.asarray(observation['footprint_world'],np.float32)
    span=float(np.linalg.norm(footprint[1]-footprint[0]))
    for index,(x,y) in enumerate(centers):
        for scale,radius in enumerate((0,1,3)):
            region=np.zeros((24,24),bool)
            region[max(0,y-radius):min(24,y+radius+1),max(0,x-radius):min(24,x+radius+1)]=True
            quality[index,scale]=float(patch[region].mean())
            masks[index,scale]=(region&patch&valid[index]).ravel()
    quality[:,3]=span/24/100
    quality[:,4]=float(support.mean())
    quality[:,5]=span/200
    return np.asarray(world,np.float32),masks,valid,quality
