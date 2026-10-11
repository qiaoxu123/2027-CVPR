"""Exactly the existing normalized A3 grid -> world transform, never retune."""
import numpy as np

def world_grid(public):
    bounds=np.asarray(public['bounds'],float);grid=np.asarray(public['grid_xy'],float)
    if bounds.shape!=(4,) or grid.ndim!=2 or grid.shape[1]!=2 or np.any(bounds[2:]<=bounds[:2]):raise ValueError('invalid original grid')
    if np.any(grid<0) or np.any(grid>1):raise ValueError('expected normalized original grid')
    return bounds[:2]+grid*(bounds[2:]-bounds[:2])
