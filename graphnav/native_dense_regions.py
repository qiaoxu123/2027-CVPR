"""Opt-in native384 dense patch lattice; ordinary observation code is unchanged."""
import numpy as np
from graphnav.landmark_projection import frame_to_world

SPARSE_IDS=np.asarray([y*24+x for y in range(1,24,2) for x in range(1,24,2)])


def native_dense_regions(observation):
    rgb=np.asarray(observation['bgr']);depth=np.asarray(observation['depth_valid'],bool)
    if rgb.shape!=(384,384,3) or depth.shape!=(384,384):raise ValueError('actual native384 RGB/depth required')
    support=(rgb.max(-1)>0)&depth;patch=support.reshape(24,16,24,16).mean((1,3))>=.9
    gy,gx=np.meshgrid(np.arange(24),np.arange(24),indexing='ij');centers=np.stack((gx.ravel(),gy.ravel()),-1)
    pixels=(centers+.5)*16-.5;world=frame_to_world(pixels,observation['footprint_world'],rgb.shape)
    masks=np.zeros((576,3,576),bool);quality=np.zeros((576,6),np.float32);valid=patch.ravel().copy()
    for i,(x,y) in enumerate(centers):
        for j,radius in enumerate((0,1,3)):
            region=np.zeros((24,24),bool);region[max(0,y-radius):min(24,y+radius+1),max(0,x-radius):min(24,x+radius+1)]=True
            quality[i,j]=patch[region].mean();masks[i,j]=(region&patch&valid[i]).ravel()
    span=float(np.linalg.norm(np.asarray(observation['footprint_world'])[1]-np.asarray(observation['footprint_world'])[0]))
    quality[:,3]=span/24/100;quality[:,4]=support.mean();quality[:,5]=span/200
    return dict(world=np.asarray(world,np.float32),pixels=pixels.astype(np.float32),masks=masks,quality=quality,valid=valid)


def sparse_view(dense):
    return {key:value[SPARSE_IDS].copy() for key,value in dense.items()}
