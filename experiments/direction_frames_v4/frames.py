"""Phase 1: semantic frames from language and public geometry only. No GT API."""
from shared import *
from infer import generate as legacy,region,anchor,immediate_name
from roads import directed_field
from shapely.geometry import Point
from shapely.ops import nearest_points
import pandas as pd

# Identifiability resolution, not an author-pose accuracy claim.
ANGLE_TOL=10.

def angular_support(a,b):
 """Angular range of B-A over two regions; includes region extent, no nearest-point oracle.
 Convex hull overapproximates nonconvex regions conservatively. Never narrows GT-wise.
 """
 av=np.array(a.convex_hull.exterior.coords[:-1]);bv=np.array(b.convex_hull.exterior.coords[:-1]);v=(bv[:,None,:]-av[None,:,:]).reshape(-1,2);aa=np.sort(np.degrees(np.arctan2(v[:,1],v[:,0]))%360)
 if a.convex_hull.intersects(b.convex_hull):return {'start_deg':0.,'width_deg':360.,'heading_deg':None,'grid_deg':list(range(360))}
 gap=np.diff(np.r_[aa,aa[0]+360]);j=int(np.argmax(gap));start=float(aa[(j+1)%len(aa)]);width=float(360-gap[j]);grid=[float(h%360) for h in np.arange(start,start+width+.1,1.)]
 return {'start_deg':start,'width_deg':width,'heading_deg':float((start+width/2)%360),'grid_deg':grid}

def scoped_text(x):
 # View cues may precede/follow relation in its sentence. An explicit stand-at
 # conditional in the next sentence is not inherited backwards.
 t=x['instruction'];s=x['direction_span'][0];bounds=[0]+[m.end() for m in re.finditer(r'[.!?]\s+',t) if not any(h['start']<=m.start()<h['end'] for h in x['mentioned'])]+[len(t)];a=max(i for i in bounds if i<=s);b=min(i for i in bounds if i>s)
 return a,b,t[a:b]

def derive(x,field_cache=None):
 cache=field_cache if field_cache is not None else {};t=x['instruction'];a,b,scope=scoped_text(x);ref=x['reference_ids'];rp=region(x,ref);r=anchor(x,ref);cat=x['scene'][ref[0]]['category'];old=legacy(x)
 p={'occurrence_id':x['occurrence_id'],'subject_ids':x['subject_ids'],'subject_role':x['subject_role'],'reference_ids':ref,'word':x['word'],'scope_span':[a,b],'frame_type':'unknown','state':'unknown','origin_xy':None,'origin_region_ids':[],'forward_heading_deg':None,'heading_support':None,'road_field_key':None,'evidence':[],'candidates':[],'reason':'no_independent_directed_frame','GT_used':False,'author_camera_pose_verified':False,'unique_definition':f'unique mathematical frame under stated geometry extraction at {ANGLE_TOL:g}deg resolution, NOT verified annotator yaw'}
 def evidence(kind,lo,hi,ids,note):p['evidence'].append({'rule':kind,'span':[lo,hi],'text':t[lo:hi],'geometry_ids':ids,'note':note})
 if rp is None or r is None:p['reason']='invalid_reference_geometry';return p
 if x['word'] in {'north','south','east','west'}:
  p.update(frame_type='world',reason='true_geographic_north_not_independently_calibrated');return p
 # The output retains geometrically constructible hypotheses but does not call
 # arbitrary axes a uniquely determined semantic frame.
 for c in old['candidates']:
  if c['type'] in {'building_PCA_axis','landmark_to_mentioned_road','road_unsigned_axis','start_to_landmark','landmark_to_start','named_pair_shared_frame_constraint'}:
   p['candidates'].append({**c,'state':'candidate','not_author_verified':True})
 # Explicit source A facing B, or looking/viewed FROM A TOWARDS B.
 observer=None
 stand=re.search(r'\bstand(?:ing)?\s+at\s+(?:the\s+)?',scope,re.I)
 if stand:
  src=immediate_name(x,a+stand.end());face=re.search(r'\bfacing\s+(?:(?:towards?|to)\s+)?',scope[stand.end():],re.I)
  if src and face:
   pos=a+stand.end()+face.end();dst=immediate_name(x,pos);ids=dst['resolved_ids'] if dst else []
   if not ids and re.match(r'(?:the\s+)?(?:stream|river|water)\b',t[pos:],re.I):
    waters=[i for i,o in x['scene'].items() if o['category']=='Water'];ids=waters if len(waters)==1 else []
   observer=(src['resolved_ids'],ids,a+stand.start(),b,'stand_at_facing')
 for fm in re.finditer(r'\b(?:looking|viewed|viewing)\s+from\s+(?:the\s+)?',scope,re.I):
  src=immediate_name(x,a+fm.end());toward=re.search(r'\b(?:towards?|at)\s+(?:the\s+)?',scope[fm.end():],re.I)
  if src and toward:
   dst=immediate_name(x,a+fm.end()+toward.end());observer=(src['resolved_ids'],dst['resolved_ids'] if dst else [],a+fm.start(),b,'looking_from_towards')
 # Facing named reference FROM named region, tightly scoped to this relation.
 for fm in re.finditer(r'\bfrom\s+(?:the\s+)?',scope,re.I):
  pre=scope[:fm.start()];src=immediate_name(x,a+fm.end());dst=None
  if not src or not re.search(r'\b(?:facing|looking|viewed)\b',pre,re.I):continue
  if re.search(r'\bfacing\s+it\s*$',pre,re.I):dst=ref
  for h in x['mentioned']:
   if a<=h['start']<h['end']<=a+fm.start() and h['resolved_ids'] and re.search(r'\b(?:facing|looking\s+at)\s+(?:the\s+)?$',t[max(a,h['start']-30):h['start']],re.I):dst=h['resolved_ids']
  if dst:observer=(src['resolved_ids'],dst,a,b,'facing_destination_from_region')
 if observer:
  src,dst,lo,hi,rule=observer;sp=region(x,src);dp=region(x,dst)
  p['frame_type']='observer';p['origin_region_ids']=src;evidence(rule,lo,hi,src+dst,'A and B are regions, not known observer/aim points; full extent retained')
  if sp is None or dp is None:p['reason']='observer_source_or_destination_not_resolved';return p
  sup=angular_support(sp,dp);p.update(heading_support=sup,origin_xy=anchor(x,src).tolist(),state='unique' if sup['width_deg']<=2*ANGLE_TOL else 'multiple',forward_heading_deg=sup['heading_deg'] if sup['width_deg']<=2*ANGLE_TOL else None,reason='region_vector_narrow_within_declared_resolution' if sup['width_deg']<=2*ANGLE_TOL else 'observer_or_aim_region_leaves_direction_range')
  p['candidates'].append({'type':'observer_region_direction_range','family':'observer','heading_support':sup,'origin_region_ids':src,'destination_ids':dst,'evidence':rule,'GT_used':False})
  return p
 # Direction given, observation origin omitted. Do not use target or nearest
 # named landmark as an invented observer position.
 if re.search(r'\b(?:when\s+facing|when\s+looking|looking\s+towards?|viewed\s+from)\b',scope,re.I):
  p.update(frame_type='observer',reason='observation_origin_or_anaphoric_destination_unknown');evidence('view_without_resolved_origin',a,b,ref,'facing a landmark alone does not locate the observer');return p
 # Road entry: relation reference itself is road, or text names an explicit road
 # being entered while comparing the target to a named building.
 roadids=ref if cat=='TrafficRoad' else None
 turn=re.search(r'\b(?:turn|take\s+a\s+(?:left|right))\s+onto\s+(?:the\s+)?',scope,re.I)
 if roadids is None and turn and x['direction_span'][0]>a+turn.end():
  h=immediate_name(x,a+turn.end())
  if h and all(x['scene'].get(i,{}).get('category')=='TrafficRoad' for i in h['resolved_ids']):roadids=h['resolved_ids']
 cms=list(re.finditer(r'\bcoming\s+from\s+(?:the\s+)?',scope,re.I))
 if roadids and cms:
  cm=cms[0];h=immediate_name(x,a+cm.end());p['frame_type']='road'
  if h and h['resolved_ids'] and not set(h['resolved_ids'])&set(roadids):
   key=x['map']+':'+','.join(roadids)+'<-'+','.join(h['resolved_ids'])
   if key not in cache:cache[key]=directed_field(x,roadids,h['resolved_ids'])
   road_mentions=[z for z in x['mentioned'] if set(z['resolved_ids'])==set(roadids) and a<=z['start']<z['end']<=b]
   if road_mentions:
    z=road_mentions[0];evidence('travel_road_A_anchor',z['start'],z['end'],roadids,'named road being entered / relation road, not nearest landmark')
   f=cache[key];p.update(road_field_key=key,state='unique_field' if f['state']=='unique_field' else 'multiple' if f['state']=='multiple' else 'unknown',origin_xy=f.get('origin_xy'),origin_region_ids=h['resolved_ids'],reason=f['reason']);evidence('road_A_entered_from_B',a+cm.start(),h['end'],roadids+h['resolved_ids'],'positive direction points away from independent source junction; complete directed centerline frozen')
   if cat!='TrafficRoad' and f['state']=='unique_field':
    # For named building reference the tangent can be anchored at this PUBLIC
    # landmark location. Target coordinates are neither read nor used.
    xy=np.array(f['points_xy']);j=int(np.argmin(np.linalg.norm(xy-r,axis=1)));p.update(state='unique',forward_heading_deg=f['heading_deg'][j],origin_xy=r.tolist(),reason='road_heading_at_independently_bound_named_reference')
   return p
  p['reason']='entry_road_not_independently_bound';return p
 if cat=='TrafficRoad':
  p['frame_type']='road';ax=shape_axis(rp)
  if ax and ax['reliable_unsigned_axis']:
   p.update(state='multiple',reason='road_axis_sign_not_specified',origin_xy=r.tolist());evidence('unsigned_road_axis',a,b,ref,'along/road location alone gives no independently known positive direction')
  return p
 if cat=='Building' and x['word'] in {'front','behind'}:
  p.update(frame_type='intrinsic',state='multiple' if any(c['type']=='building_PCA_axis' for c in p['candidates']) else 'unknown',origin_xy=r.tolist(),reason='no_independent_entrance_or_directed_facade');evidence('intrinsic_front_not_verified',x['direction_span'][0],x['reference_span'][1],ref,'PCA is unsigned and does not identify facade, entrance, or author view')
 # Named-to-named shared image view constraints remain a family of permissible
 # views, not an exact yaw recovered from a binary left/right statement.
 shared=[c for c in p['candidates'] if c['type']=='named_pair_shared_frame_constraint']
 if shared and re.search(r'\b(?:aerial|aerially|viewed|viewing)\b',scope,re.I):
  p.update(frame_type='observer',state='multiple',reason='shared_named_landmark_view_is_a_constraint_not_exact_heading');evidence('shared_view_constraint',a,b,[i for c in shared for i in c['source_ids']],'assuming one view within this clause; binary directional constraints do not identify an exact angle')
 return p

def run():
 X=readlines(OUT/'public_inputs.jsonl.gz');cache={};P=[derive(x,cache) for x in X];by={p['occurrence_id']:p for p in P};R=readlines(OUT/'relations.jsonl.gz')
 for r in R:
  if r['occurrence_id'] not in by:P.append({'occurrence_id':r['occurrence_id'],'subject_ids':r.get('subject_ids',[]),'reference_ids':r.get('reference_ids',[]),'word':r['semantic_word'],'frame_type':'unknown','state':'unknown','forward_heading_deg':None,'origin_xy':None,'evidence':[],'candidates':[],'GT_used':False,'reason':'binding_or_scope:'+r['reason']})
 for p in P:
  p['uniquely_determined_geometry_frame']=p['state'] in {'unique','unique_field'};p['unique_scalar_heading']=p['state']=='unique';p['frame_direction_representation']='frozen spatial road field' if p['state']=='unique_field' else 'angular range or candidate set' if p['state']=='multiple' else 'unknown' if p['state']=='unknown' else 'scalar heading'
 writelines(OUT/'frozen_frames.jsonl.gz',P);writelines(OUT/'road_fields.jsonl.gz',[{'key':k,**v} for k,v in cache.items()]);pd.DataFrame([{k:v for k,v in p.items() if k not in {'candidates','heading_support'}} for p in P]).to_csv(OUT/'frame_evidence.csv',index=False)
 writej(OUT/'freeze_manifest.json',{'phase':1,'inputs_sha256':sha(OUT/'public_inputs.jsonl.gz'),'frame_predictions_sha256':sha(OUT/'frozen_frames.jsonl.gz'),'road_fields_sha256':sha(OUT/'road_fields.jsonl.gz'),'code_sha256':{f:sha(BASE4/f) for f in ['frames.py','roads.py','road_language.py','audit.py','shared.py']},'bound_states':dict(collections.Counter(p['state'] for p in P if p['occurrence_id'] in {x['occurrence_id'] for x in X})),'all_states':dict(collections.Counter(p['state'] for p in P)),'no_target_GT_or_future_input':True,'angular_resolution_deg':ANGLE_TOL,'author_pose_verified':False})
 print(readj(OUT/'freeze_manifest.json'))
if __name__=='__main__':run()
