"""Exact simulator-only frustum culling; no raw-cloud information to policy."""
import numpy as np
from experiments.multiview_rgb_grounding.render import basis,render

def frustum_bounds(pose,tilt,z_bounds):
    # Axial far400 contains the original renderer's Euclidean far400 sphere.
    v=np.array([[x*z,y*z,z] for z in [.5,400.] for x in [-1,1] for y in [-1,1]])@basis(pose[3],tilt)+np.asarray(pose[:3])
    lo,hi=z_bounds;points=[x for x in v if lo<=x[2]<=hi]
    pairs=[(0,1),(0,2),(1,3),(2,3),(4,5),(4,6),(5,7),(6,7)]+[(i,i+4) for i in range(4)]
    for a,b in pairs:
        if abs(v[a,2]-v[b,2])<1e-12:continue
        for level in [lo,hi]:
            t=(level-v[a,2])/(v[b,2]-v[a,2])
            if 0<=t<=1:points.append(v[a]+t*(v[b]-v[a]))
    if not points:return None
    p=np.asarray(points);return p.min(0)[:2]-.001,p.max(0)[:2]+.001

def render_culled(xyz,rgb,pose,tilt,z_bounds=None):
    z_bounds=z_bounds if z_bounds is not None else (float(xyz[:,2].min()),float(xyz[:,2].max()))
    bounds=frustum_bounds(pose,tilt,z_bounds)
    if bounds is None:ids=np.zeros(0,np.int64)
    else:
        lo,hi=bounds;ids=np.flatnonzero((xyz[:,0]>=lo[0])&(xyz[:,0]<=hi[0])&(xyz[:,1]>=lo[1])&(xyz[:,1]<=hi[1]))
    f=render(xyz[ids],rgb[ids],pose,tilt,radius=1);win=f['winner'];valid=win>=0;win[valid]=ids[win[valid]];f['input_points_after_exact_culling']=len(ids)
    return f
