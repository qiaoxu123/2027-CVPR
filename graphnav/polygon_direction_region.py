"""TRAIN experiment: public Polygon front-envelope direction regions.

The compass names retain the engineering +Y/+X convention, not a verified
geographic interpretation. No target data, RGB or learned weights are used.
"""
from __future__ import annotations
import numpy as np
import shapely
from shapely.geometry import Polygon,LineString,box
from shapely.ops import unary_union

UNITS={'north':np.array([0.,1.]),'south':np.array([0.,-1.]),'east':np.array([1.,0.]),'west':np.array([-1.,0.])}

def validate_public_record(row):
 if {'target_world','target_ids','target_type','target_positions','object_ids'}&set(row):
  raise ValueError('Target supervision is forbidden in direction inference records')

def polygon_direction_region(polygons,bounds,word,mode='cone'):
 """Return 28x28 public directional probability/support and availability.

 ray: extend the full directional silhouette, independently at each grid line.
 cone: extend its sampled front envelope with a fixed 45-degree opening.
 Distances remain continuous with a heavy-tailed prior, not a fixed translation.
 Empty support is an explicit abstention, not a fabricated location.
 """
 if word not in UNITS or mode not in ('ray','cone'):raise ValueError('Known cardinal word and ray/cone required')
 b=np.asarray(bounds,float)
 if b.shape!=(4,) or not np.isfinite(b).all() or (b[2:]<=b[:2]).any():raise ValueError('Finite ascending map bounds required')
 pieces=[]
 for xy in polygons:
  xy=np.asarray(xy,float)
  if xy.ndim!=2 or xy.shape[1]!=2 or len(xy)<3 or not np.isfinite(xy).all():raise ValueError('Finite public XY Polygon required')
  p=Polygon(xy)
  if not p.is_valid or p.area<=0:raise ValueError('Invalid public Polygon; do not silently repair its identity')
  pieces.append(p)
 if not pieces:raise ValueError('At least one public Polygon required')
 region=unary_union(pieces).intersection(box(*b));axis=(np.arange(28)+.5)/28;yy,xx=np.meshgrid(axis,axis,indexing='ij');grid=np.stack([xx,yy],-1)*(b[2:]-b[:2])+b[:2];points=grid.reshape(-1,2);u=UNITS[word];v=np.array([-u[1],u[0]]);along=points@u;cross=points@v;values=np.zeros(len(points));inside=shapely.covers(region,shapely.points(points))
 scale=max(20.,np.sqrt(region.area)/2);L=max(np.abs(b).max(),np.abs(shapely.get_coordinates(region)).max(initial=0))+2*max(b[2:]-b[:2])+1
 def front(t):
  coords=shapely.get_coordinates(region.intersection(LineString([v*t-u*L,v*t+u*L])))
  return None if not len(coords) else float((coords@u).max())
 if mode=='ray':
  for t in np.unique(cross):
   f=front(t)
   if f is None:continue
   distance=along-f;valid=np.isclose(cross,t,atol=1e-10,rtol=0)&(distance>1e-8)&~inside
   values[valid]=1/(1+(distance[valid]/scale)**2)
 else:
  ts=[]
  geoms=list(region.geoms) if hasattr(region,'geoms') else [region]
  for g in geoms:
   if g.is_empty:continue
   cc=shapely.get_coordinates(g)@v;lo,hi=cc.min(),cc.max()
   ts.extend(lo+(np.arange(64)+.5)/64*(hi-lo))
  for t in np.unique(ts):
   f=front(t)
   if f is None:continue
   d=along-f;lateral=np.abs(cross-t);valid=(d>1e-8)&(lateral<=d)&~inside
   distance2=d[valid]**2+lateral[valid]**2;cosine2=d[valid]**2/np.maximum(distance2,1e-20);weight=cosine2/(1+distance2/scale**2)
   values[valid]=np.maximum(values[valid],weight)
 available=bool(values.sum()>0)
 if available:values/=values.sum()
 return dict(probabilities=values.reshape(28,28).astype(np.float32),support=(values>0).reshape(28,28),available=available,reason='public_direction_region' if available else 'no_directional_grid_support',world_grid=grid,scale_m=scale,mode=mode)
