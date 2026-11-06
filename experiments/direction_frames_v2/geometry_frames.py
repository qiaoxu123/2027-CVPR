"""Offline geometry and candidate frames, AFTER immutable text binding.
No per-target best-sign choice. Future/target-conditioned sources are explicit.
"""
from __future__ import annotations
import math
import numpy as np
from shapely.geometry import Polygon,Point
from shapely.ops import unary_union,nearest_points

OFFSETS={'front':0.,'left':90.,'behind':180.,'back':180.,'rear':180.,'right':270.}
IMAGE={'left':np.array([-1.,0.]),'right':np.array([1.,0.]),'front':np.array([0.,1.]),'behind':np.array([0.,-1.]),'back':np.array([0.,-1.]),'rear':np.array([0.,-1.]),'above':np.array([0.,1.]),'below':np.array([0.,-1.]),'top':np.array([0.,1.]),'bottom':np.array([0.,-1.])}

def angle(v):return float(np.degrees(np.arctan2(v[1],v[0]))%360)
def adiff(a,b):return np.abs((np.asarray(a)-b+180)%360-180)
def poly(o):
 if len(o.get('contour',[]))<3:return None
 p=Polygon(o['contour']);return p if p.is_valid and p.area>0 else None

def vertices(p):
 parts=list(p.geoms) if p.geom_type=='MultiPolygon' else [p]
 return np.vstack([np.asarray(q.exterior.coords) for q in parts])

def shape_axis(p):
 """PCA on uniformly sampled exterior arc length, not vertex density."""
 if p is None or p.length<=0:return None
 parts=list(p.geoms) if p.geom_type=='MultiPolygon' else [p]
 # Fixed total perimeter sample allocation; reproducible and independent of target.
 arr=[]
 for q in parts:
  n=max(12,int(round(256*q.length/p.length)))
  arr.extend(q.exterior.interpolate(d).coords[0] for d in np.linspace(0,q.exterior.length,n,endpoint=False))
 x=np.array(arr);cov=np.cov(x.T);ev,vec=np.linalg.eigh(cov);v=vec[:,np.argmax(ev)]
 ratio=float(max(ev)/max(min(ev),1e-9));return {'axis_deg':angle(v)%180,'eigenvalue_ratio':ratio,'reliable_unsigned_axis':ratio>=2.,'method':'uniform_perimeter_PCA_256','directed_heading_verified':False}

def transforms():
 out=[]
 for mir,label in [(np.eye(2),'none'),(np.diag([-1.,1.]),'X'),(np.diag([1.,-1.]),'Y')]:
  for rot in [0,90,180,270]:
   a=np.deg2rad(rot);m=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])@mir
   out.append({'name':f'A_image_rot{rot}_mirror{label}','matrix':np.rint(m).astype(int).tolist(),'rotation_deg':rot,'mirror':label,'future':False,'target_conditioned':False,'heading_verified':False,'interpretation':'fixed_global_image_dictionary_candidate'})
 return out

def vector_heading(v,minimum=1.):return angle(v) if np.linalg.norm(v)>minimum else None

def initial_motion(path,min_displacement=3.):
 p=np.asarray(path,float)[:,:2]
 for j in range(1,len(p)):
  if np.linalg.norm(p[j]-p[0])>=min_displacement:return angle(p[j]-p[0]),[0,j]
 return None,None

def approach_motion(path,k,min_displacement=3.):
 p=np.asarray(path,float)[:,:2]
 if k<=0:return None,None
 for j in range(k-1,-1,-1):
  if np.linalg.norm(p[k]-p[j])>=min_displacement:return angle(p[k]-p[j]),[j,k]
 return None,None

def boundary_class(tpoly,rpoly,u,tol=1.):
 t=vertices(tpoly)@u;r=vertices(rpoly)@u;gap=float(min(t)-max(r));opp=float(max(t)-min(r))
 return ('consistent' if gap>tol else 'opposite' if opp< -tol else 'ambiguous'),gap,opp

def road_tangent_at_start(rpoly,start):
 """Boundary tangent near independent start point; not target-nearest tangent.
 Endpoint/end-cap ambiguity is not silently treated as a verified road centerline.
 """
 if rpoly is None:return None
 parts=list(rpoly.geoms) if rpoly.geom_type=='MultiPolygon' else [rpoly]
 # Segment choice is start-conditioned, never target-conditioned; binding stays whole entity.
 q=min(parts,key=lambda p:p.distance(Point(*start)));ring=q.exterior;pt=nearest_points(Point(*start),ring)[1];s=ring.project(pt);L=ring.length
 a=np.array(ring.interpolate((s-5)%L).coords[0]);b=np.array(ring.interpolate((s+5)%L).coords[0]);v=b-a
 if np.linalg.norm(v)<3:return None
 return angle(v)%180

def compute_pair(row,ep,objects):
 subject_ids=row.get('subject_ids') or [row['target_id']]
 if len(subject_ids)!=1:raise ValueError('composite relation subject requires independent geometry support')
 t=objects[subject_ids[0]];refs=[objects[i] for i in row['reference_ids']];tp=poly(t);ps=[poly(o) for o in refs]
 rp=unary_union(ps) if all(p is not None for p in ps) else None
 # Single-object original position preserves v1 comparison; multi-road uses ALL geometry.
 rxy=np.array(refs[0]['position'][:2],float) if len(refs)==1 else np.array(rp.centroid.coords[0]) if rp is not None else np.mean([o['position'][:2] for o in refs],axis=0)
 gt=np.array(ep['target_positions'][0] if row.get('subject_role','annotated_target')=='annotated_target' else t['position'],float);d=gt[:2]-rxy;path=ep['trajectory'];start=np.array(path[0][:2]);poses=np.asarray(path,float);refcat=refs[0]['object_type']
 pair={**row,'cluster':f"{row['map']}:{subject_ids[0]}",'relation_subject_id':subject_ids[0],'relation_subject_category':t['object_type'],'target_xy':gt[:2].tolist(),'reference_xy':rxy.tolist(),'gt_xyz':gt.tolist(),'dx':float(d[0]),'dy':float(d[1]),'angle_world':angle(d),'distance_m':float(np.linalg.norm(d)),'reference_category':refcat,'reference_anchor':'original_position' if len(refs)==1 else 'centroid_of_all_same_named_road_polygons','target_polygon_valid':tp is not None,'reference_polygon_valid':rp is not None,'geometry_status':'invalid_polygon','reference_heading_verified':False,'initial_pose':path[0]}
 frames={}
 def frame(name,h,kind,window=None,future=False,conditioned=False,verified=False,note=''):
  frames[name]={'heading_deg':h,'kind':kind,'window':window,'future':future,'target_conditioned':conditioned,'heading_verified':verified,'note':note}
 yaw=vector_heading(poses[0,3:5],1e-6) if poses.shape[1]==6 else float(np.degrees(poses[0,3])%360)
 los=vector_heading(rxy-start)
 frame('B_start_to_reference',los,'start_reference_line')
 frame('B_reference_to_start',(los+180)%360 if los is not None else None,'reversed_start_reference_line')
 frame('C_initial_yaw',yaw,'initial_look_vector',future=False,note='navigation initial yaw, not annotation camera yaw')
 move,win=initial_motion(path);frame('C_initial_actual_motion',move,'first_3m_horizontal_displacement',window=win,future=True)
 kref=None
 if tp is not None and rp is not None:
  rdist=np.array([rp.distance(Point(*p)) for p in poses[:,:2]]);kref=int(np.argmin(rdist));h,win=approach_motion(path,kref)
  frame('C_motion_approaching_reference',h,'incoming_motion_before_first_closest_reference_polygon',window=win,future=True,note='nearest event chosen relative to already text-bound reference polygon')
  kt=int(np.argmin(np.linalg.norm(poses[:,:2]-np.array(ep['target_positions'][0][:2]),axis=1)));h,win=approach_motion(path,kt)
  frame('C_motion_approaching_target',h,'incoming_motion_before_first_closest_GT_target',window=win,future=True,conditioned=True,note='GT-conditioned offline diagnostic; not independent author-frame evidence')
  pair.update(polygon_distance_m=float(tp.distance(rp)),intersection_area=float(tp.intersection(rp).area),target_area_m2=float(tp.area),reference_area_m2=float(rp.area),target_gt_in_polygon=tp.buffer(1).covers(Point(*gt[:2])),target_gt_vs_object_m=float(np.linalg.norm(gt[:2]-np.array(t['position'][:2]))),reference_closest_step=kref,target_closest_step=kt)
  pair['geometry_status']='disjoint' if not tp.intersects(rp) and tp.distance(rp)>1 and np.linalg.norm(d)>1 else 'ambiguous_overlap_or_boundary'
  pair['polygon_centroid_angle']=angle(np.array(tp.centroid.coords[0])-np.array(rp.centroid.coords[0]))
  q=nearest_points(rp,Point(*gt[:2]))[0];v=gt[:2]-np.array(q.coords[0]);pair['nearest_reference_boundary_angle']=vector_heading(v)
  ax=shape_axis(rp);pair['reference_unsigned_axis']=ax
  if ax and ax['reliable_unsigned_axis']:
   a=ax['axis_deg']
   if refcat=='TrafficRoad':
    for delta in [0,180]:frame(f'D_road_axis_sign{delta}',(a+delta)%360,'unsigned_road_shape_axis',note='fixed arbitrary sign; not verified travel direction')
    if move is not None:
     h=a if adiff(a,move)<=90 else (a+180)%360;frame('D_road_axis_aligned_initial_motion',h,'road_axis_signed_by_initial_motion',window=frames['C_initial_actual_motion']['window'],future=True,note='navigation-motion proxy, not independently verified road direction')
    tangent=road_tangent_at_start(rp,start)
    if tangent is not None:
     for delta in [0,180]:frame(f'D_road_local_tangent_start_sign{delta}',(tangent+delta)%360,'road_boundary_tangent_at_start',note='boundary tangent proxy; may hit end caps; fixed sign')
   elif refcat=='Building':
    for delta in [0,90,180,270]:frame(f'D_building_axis_offset{delta}',(a+delta)%360,'unsigned_building_shape_axis',note='PCA axis/normal candidates, no facade or entrance heading')
  if row.get('binding_rule')=='explicit_stand_at_facing_frame' and row.get('declared_facing_category') in {'river','stream'}:
   waters=[(i,o,poly(o)) for i,o in objects.items() if o['object_type']=='Water']
   pair['declared_water_candidate_ids']=[i for i,o,p in waters]
   if len(waters)==1 and waters[0][2] is not None:
    wpoly=waters[0][2];q=nearest_points(Point(*rxy),wpoly)[1];h=vector_heading(np.array(q.coords[0])-rxy)
    frame('D_explicit_stand_facing_water',h,'text_declared_viewpoint_unique_water_boundary',verified=False,note='semantic facing stream; unique Water polygon; nearest water point uses reference position only, never target GT')
   else:frame('D_explicit_stand_facing_water',None,'unresolved_water_identity',note='multiple/missing water objects; no GT disambiguation')
 pair['frames']=frames
 pair['D_unavailable']={'building_directed_facade':'no entrance/facade heading annotation','car_directed_heading':'no independently text-bound named car reference','parking_lane':'no named Parking instances or independently identified lane geometry'}
 return pair,tp,rp

def evaluate(pair,tp,rp,frame_name,frame,word=None,tol=1.):
 w=word or pair['semantic_word'];h=frame.get('heading_deg');u=None
 if 'matrix' in frame:
  if w in IMAGE:u=np.array(frame['matrix'])@IMAGE[w]
 elif w in OFFSETS and h is not None:
  a=np.deg2rad(h+OFFSETS[w]);u=np.array([np.cos(a),np.sin(a)])
 if u is None:return None
 expected=angle(u);err=float(adiff(pair['angle_world'],expected));cl,gap,opp=boundary_class(tp,rp,u,tol) if tp is not None and rp is not None else ('unavailable',None,None)
 return {'occurrence_id':pair['occurrence_id'],'cluster':pair['cluster'],'map':pair['map'],'word':w,'hypothesis':frame_name,'parse_tier':pair['confidence_tier'],'parse_confidence':pair['parse_confidence'],'geometry_status':pair['geometry_status'],'reference_category':pair['reference_category'],'target_category':pair['relation_subject_category'],'expected_world_deg':expected,'angle_error_deg':err,'sector_correct':bool(err<=45),'halfplane_correct':bool(np.dot(np.array([pair['dx'],pair['dy']]),u)>tol),'polygon_class':cl,'projection_gap_m':gap,'opposite_gap_m':opp,'future':frame.get('future',False),'target_conditioned':frame.get('target_conditioned',False),'heading_verified':frame.get('heading_verified',False),'explicit_view':pair['explicit_view'],'local_layout':pair['local_layout']}
