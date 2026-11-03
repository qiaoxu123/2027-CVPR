"""Diagnostic public observation sites; normal candidate sampling unchanged."""
import numpy as np
from .landmark_projection import frame_to_world


def observed_grid_sites(observation, stride):
    if stride not in (1,2): raise ValueError('native or existing grid only')
    rgb=np.asarray(observation['bgr']); depth=np.asarray(observation['depth_valid'],bool)
    if rgb.shape!=(384,384,3) or depth.shape!=(384,384): raise ValueError('native current 384px observation required')
    support=(rgb.max(-1)>0)&depth
    patch=support.reshape(24,16,24,16).mean((1,3))>=.9
    gy,gx=np.meshgrid(np.arange(stride-1,24,stride),np.arange(stride-1,24,stride),indexing='ij')
    centers=np.stack((gx.ravel(),gy.ravel()),-1)
    pixels=(centers+.5)*16-.5
    return frame_to_world(pixels,observation['footprint_world'],rgb.shape),patch[centers[:,1],centers[:,0]]
