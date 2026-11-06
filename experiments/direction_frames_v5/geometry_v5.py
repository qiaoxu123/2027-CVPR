"""Identical full-contour predicate and projection for all orientation controls."""
from v5core import *
from shapely.geometry import Polygon,LineString,Point
from shapely.ops import unary_union

def projection(f,contour,tol=1.):
 xy=np.array(f['points_xy']);ed=np.array(f['edges']);a=xy[ed[:,0]];v=xy[ed[:,1]]-a;ll=(v*v).sum(1);h=np.deg2rad(np.array(f['heading_deg'],float));unit=np.stack([np.cos(h),np.sin(h)],1);u=unit[ed[:,0]]+unit[ed[:,1]];length=np.linalg.norm(u,axis=1);valid=(ll>1e-12)&np.isfinite(length)&(length>1e-12)
 # Smoothing can contract consecutive graph nodes to the same coordinate.
 # Ignore mathematically zero-length segments without choosing target-specific edges.
 ed=ed[valid];a=a[valid];v=v[valid];ll=ll[valid];u=u[valid]/length[valid,None]
 if not len(ed):return {'mask_reason':'degenerate_centerline_geometry','pts':np.zeros((0,2)),'nearest':np.zeros((0,2)),'base_unit':np.zeros((0,2)),'line':LineString(),'distance_to_centerline_m':None}
 pts=boundary_samples(contour);rel=pts[:,None]-a;z=np.clip((rel*v).sum(2)/ll,0,1);qq=a+z[:,:,None]*v;dist=((pts[:,None]-qq)**2).sum(2);ix=dist.argmin(1);near=qq[np.arange(len(pts)),ix];base=u[ix];ties=dist<=dist.min(1)[:,None]+.25;reason=None
 for i in range(len(pts)):
  jj=np.flatnonzero(ties[i]);angles=np.degrees(np.arctan2(u[jj,1],u[jj,0]));ref=np.degrees(np.arctan2(base[i,1],base[i,0]));
  if len(jj)>1 and np.max(adiff(angles,ref))>30:reason='multiple_nearest_tangents';break
 line=unary_union([LineString([xy[i],xy[j]]) for i,j in ed]);target=Polygon(contour)
 if reason is None and target.intersects(line.buffer(tol)):reason='target_contour_touches_centerline_tolerance'
 return {'pts':pts,'nearest':near,'base_unit':base,'mask_reason':reason,'line':line,'distance_to_centerline_m':float(target.distance(line))}

def orientation_score(pr,word,heading=None,rotation=0.,tol=1.):
 if pr['mask_reason']:return {'result':'ambiguous','score':0.,'reason':pr['mask_reason']}
 if heading is None:
  r=np.deg2rad(rotation);u=pr['base_unit']@np.array([[np.cos(r),np.sin(r)],[-np.sin(r),np.cos(r)]])
 else:
  h=np.deg2rad(heading);u=np.repeat([[np.cos(h),np.sin(h)]],len(pr['pts']),axis=0)
 normal=np.stack([-u[:,1],u[:,0]],1);z=((pr['pts']-pr['nearest'])*normal).sum(1);z=z if word=='left' else -z;result='consistent' if min(z)>tol else 'conflict' if max(z)<-tol else 'ambiguous'
 return {'result':result,'score':float(result=='consistent'),'reason':'identical_frozen_centerline_full_contour_side','margin_m':float(min(z)),'opposite_margin_m':float(max(z))}

def point_side(f,xy,word):
 points=np.array(f['points_xy']);i=int(np.argmin(((points-np.array(xy))**2).sum(1)));h=np.deg2rad(f['heading_deg'][i]);v=np.array(xy)-points[i];s=float(v@np.array([-np.sin(h),np.cos(h)]));s=s if word=='left' else -s
 return {'signed_margin_m':s,'result':'consistent' if s>1 else 'conflict' if s< -1 else 'ambiguous'}
