"""Efficient EXACT centroid/FOV association, verified against frozen P0 code.

Only one cell is inspected rather than constructing 784 polygons for each
region. This is a performance change; original grid and support rule preserved.
"""
import numpy as np
from shapely.geometry import Polygon,box
from graphnav.memory import ObservedRGBMemory

def centroid_support(contour,footprint,bounds,grid_xy):
    poly=Polygon(np.asarray(contour,float));view=Polygon(np.asarray(footprint,float))
    if not poly.is_valid or not view.is_valid or poly.area<=0 or view.area<=0:raise ValueError('valid finite public polygons required')
    result=np.zeros(784,float)
    if poly.intersection(view).area<=1e-8:return result
    bounds=np.asarray(bounds,float);grid=np.asarray(grid_xy,float)
    if bounds.shape!=(4,) or grid.shape!=(784,2) or np.any(bounds[2:]<=bounds[:2]):raise ValueError('unchanged grid required')
    world=bounds[:2]+grid*(bounds[2:]-bounds[:2]);center=np.asarray(contour,float).mean(0);index=int(np.square(world-center).sum(1).argmin());half=(bounds[2:]-bounds[:2])/56
    cell=box(world[index,0]-half[0],world[index,1]-half[1],world[index,0]+half[0],world[index,1]+half[1])
    if cell.intersection(view).area>1e-8:result[index]=1
    return result

def weights(state,regions):
    result=[]
    for v in regions:
        pose=np.asarray(v['actual_camera_pose']);height=pose[2]-v['ground_level'];foot=ObservedRGBMemory.view_corners(pose[:2],pose[3],height)
        result.append(centroid_support(v['contour'],foot,state['public']['bounds'],state['public']['grid_xy']))
    return np.asarray(result).reshape(len(regions),784)
