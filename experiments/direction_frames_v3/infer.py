"""Reference frames from TEXT + named-landmark geometry only. No GT/trajectory API."""
from dataclasses import dataclass
import re,collections
from shapely.geometry import Polygon,Point
from shapely.ops import unary_union,nearest_points
from common import *

FAMILIES=['observer','road_travel','image_layout','intrinsic_candidate','unknown']
def object_poly(o):
 if len(o['contour'])<3:return None
 p=Polygon(o['contour']);return p if p.is_valid and p.area>0 else None

def region(inp,ids):
 if not ids or any(i not in inp['scene'] for i in ids):return None
 ps=[object_poly(inp['scene'][i]) for i in ids]
 if any(p is None for p in ps):return None
 return unary_union(ps)

def anchor(inp,ids):
 p=region(inp,ids)
 if p is None:return None
 return np.array(inp['scene'][ids[0]]['position'],float) if len(ids)==1 else np.array(p.centroid.coords[0])

def immediate_name(inp,start):
 hits=[h for h in inp['mentioned'] if h['start']>=start and re.fullmatch(r'\s*(?:the\s+)?',inp['instruction'][start:h['start']],re.I)]
 return hits[0] if len(hits)==1 and hits[0]['resolved_ids'] else None

def language_family(inp):
 text=inp['instruction'];refs=[inp['scene'][i] for i in inp['reference_ids'] if i in inp['scene']];cat=refs[0]['category'] if refs else 'unknown'
 if re.search(r'\bstand(?:ing)?\s+at\b|\b(?:when|if)\b.{0,35}\b(?:looking|facing|viewing)\b.{0,100}\bfrom\b|\bviewed\s+from\b',text,re.I):return 'observer'
 if cat=='TrafficRoad' and re.search(r'\bcoming\s+from\b',text,re.I):return 'road_travel'
 if re.search(r'\b(?:viewed|viewing|aerial|aerially|with)\b.{0,180}\b(?:horizontal|vertical|orientated|oriented)|\bwhen\s+viewed\s+aerially\b',text,re.I):return 'image_layout'
 if cat=='Building' and inp['word'] in {'front','behind'}:return 'intrinsic_candidate'
 return 'unknown'

def generate(inp):
 text=inp['instruction'];refs=inp['reference_ids'];rp=region(inp,refs);r=anchor(inp,refs);out=[];issues=[]
 if r is None:return {'occurrence_id':inp['occurrence_id'],'family_silver':language_family(inp),'candidates':[],'strict_state':'unknown','strict_ids':[],'unknown_heading':True,'reasons':['invalid_reference_polygon'],'reference_category':'unknown'}
 cat=inp['scene'][refs[0]]['category'];pca=shape_axis(rp)
 def add(kind,h,family,level,evidence,source_ids,confidence=.5,origin=None,meta=None):
  if h is None:return
  cid=f'{kind}:{len(out)}';out.append({'candidate_id':cid,'type':kind,'heading_deg':float(h%360),'family':family,'level':level,'evidence':evidence,'source_ids':source_ids,'confidence_rule_score':confidence,'origin_xy':(r if origin is None else np.asarray(origin)).tolist(),'meta':meta or {},'GT_used_to_construct_or_select':False})
 def vector(kind,v,family,level,evidence,source_ids,confidence=.5,origin=None,meta=None):
  if np.linalg.norm(v)>1:add(kind,angle(v),family,level,evidence,source_ids,confidence,origin,meta)
 # Baseline local line hypotheses remain candidates, not default author frames.
 start=np.array(inp['start_xy']);vector('landmark_to_start',start-r,'unknown','candidate','episode start is known; no language authorization of author frame',refs,.2)
 vector('start_to_landmark',r-start,'unknown','candidate','episode start is known; no language authorization of author frame',refs,.2)
 others=[];road_mentions=[]
 for hit in inp['mentioned']:
  ids=hit['resolved_ids']
  if not ids or set(ids)&set(refs) or any(i not in inp['scene'] for i in ids):continue
  if tuple(ids) not in [tuple(x['resolved_ids']) for x in others]:others.append(hit)
  if all(inp['scene'][i]['category']=='TrafficRoad' for i in ids):road_mentions.append(hit)
 for h in others:
  a=anchor(inp,h['resolved_ids'])
  if a is not None:vector('landmark_to_other_named',a-r,'unknown','candidate','other named entity mentioned; its role does not establish a view direction',refs+h['resolved_ids'],.25)
 for h in road_mentions:
  q=region(inp,h['resolved_ids'])
  if q is not None:
   near=nearest_points(Point(*r),q)[1];v=np.array(near.coords[0])-r
   vector('landmark_to_mentioned_road',v,'intrinsic_candidate' if cat=='Building' else 'unknown','candidate','road-facing facade hypothesis; facade/entrance NOT independently verified',refs+h['resolved_ids'],.45)
 if pca and pca['reliable_unsigned_axis']:
  if cat=='Building':
   for delta in [0,90,180,270]:add('building_PCA_axis',pca['axis_deg']+delta,'intrinsic_candidate','candidate','unsigned building axis/normal; no target-dependent sign choice',refs,.3,meta={'offset_deg':delta,'axis_ratio':pca['eigenvalue_ratio']})
  if cat=='TrafficRoad':
   for delta in [0,180]:add('road_unsigned_axis',pca['axis_deg']+delta,'road_travel' if re.search(r'coming\s+from',text,re.I) else 'unknown','candidate','road axis has no positive travel direction',refs,.35,meta={'offset_deg':delta,'axis_ratio':pca['eigenvalue_ratio']})
 # Explicit named viewpoint: stand at NAME facing REGION/NAME.
 stand=re.search(r'\bstand(?:ing)?\s+at\s+(?:the\s+)?',text,re.I)
 if stand:
  sh=immediate_name(inp,stand.end());face=re.search(r'\bfacing\s+(?:(?:towards?|to)\s+)?',text[stand.end():],re.I)
  if sh and face:
   pos=stand.end()+face.end();src=anchor(inp,sh['resolved_ids']);dest=immediate_name(inp,pos)
   if src is not None and dest:
    dst=anchor(inp,dest['resolved_ids'])
    if dst is not None:vector('explicit_named_facing',dst-src,'observer','explicit','stand at named source facing named destination',sh['resolved_ids']+dest['resolved_ids'],.95)
   elif src is not None and re.match(r'(?:the\s+)?(?:stream|river|water)\b',text[pos:],re.I):
    water=[i for i,o in inp['scene'].items() if o['category']=='Water']
    if len(water)==1 and region(inp,water) is not None:
     wp=region(inp,water);q=nearest_points(Point(*src),wp)[1];v=np.array(q.coords[0])-src
     # Deterministic nearest-bank-normal proxy, assessed at source contour vertices.
     sp=region(inp,sh['resolved_ids']);hs=[]
     for xy in vertices(sp):
      qq=nearest_points(Point(*xy),wp)[1];vv=np.array(qq.coords[0])-xy
      if np.linalg.norm(vv)>1:hs.append(angle(vv))
     spread=max([float(adiff(h,angle(v))) for h in hs],default=180.)
     vector('explicit_stand_face_water',v,'observer','explicit' if spread<=30 else 'candidate','stand at named landmark facing unique water; nearest-bank proxy, not recovered camera yaw',sh['resolved_ids']+water,.93,meta={'source_perturbation_spread_deg':spread,'water_boundary_point':list(q.coords[0])})
    else:issues.append('water_not_uniquely_identified')
 # Observer facing primary landmark FROM an explicit named region.
 for fm in re.finditer(r'\bfrom\s+(?:the\s+)?',text,re.I):
  pre=text[max(0,fm.start()-140):fm.start()]
  if not re.search(r'\b(?:facing|look(?:ing)?|viewed|viewing)\b',pre,re.I):continue
  src_hit=immediate_name(inp,fm.end())
  if src_hit is None:continue
  # Require a directly named destination or tightly licensed 'it' referring to reference.
  dest_ids=None;approx=False
  for h in inp['mentioned']:
   if h['end']<=fm.start() and h['start']>=max(0,fm.start()-130) and h['resolved_ids'] and re.search(r'\b(?:facing|looking\s+at|viewing)\s+(?:the\s+)?$',text[max(0,h['start']-30):h['start']],re.I):dest_ids=h['resolved_ids']
  if dest_ids is None and re.search(r'\bfacing\s+it\s*$',pre,re.I):
   rr=inp.get('reference_span');later=[h for h in inp['mentioned'] if rr and rr[1]<h['start']<fm.start()]
   if rr and rr[1]<fm.start() and not later:dest_ids=refs
  if dest_ids is None and re.search(r'\blooking\s+at\s+them\s*$',pre,re.I):dest_ids=refs;approx=True
  if dest_ids is None:continue
  dst=anchor(inp,dest_ids);srcp=region(inp,src_hit['resolved_ids'])
  if dst is None or srcp is None:continue
  origin=np.array(nearest_points(Point(*dst),srcp)[1].coords[0]);v=dst-origin
  if np.linalg.norm(v)<=1:continue
  spread=max(float(adiff(angle(dst-x),angle(v))) for x in vertices(srcp) if np.linalg.norm(dst-x)>1)
  lvl='explicit' if spread<=30 and not approx else 'candidate'
  vector('explicit_view_from_named_region',v,'observer',lvl,'language specifies observer source region; nearest point is a representative and region extent may leave heading ambiguous',src_hit['resolved_ids']+dest_ids,.9 if lvl=='explicit' else .6,meta={'source_region_spread_deg':spread,'observer_source_xy':origin.tolist(),'group_gaze_approximation':approx})
 # Directed travel along named road FROM another named road. Sign depends ONLY on
 # junction topology / road shape, never the target's side or position.
 if cat=='TrafficRoad' and pca and pca['eigenvalue_ratio']>=8:
  for cm in re.finditer(r'\bcoming\s+from\s+(?:the\s+)?',text,re.I):
   h=immediate_name(inp,cm.end())
   if h is None or set(h['resolved_ids'])&set(refs):continue
   source=region(inp,h['resolved_ids'])
   if source is None:continue
   inter=rp.intersection(source);gap=rp.distance(source)
   q=np.array(inter.centroid.coords[0]) if not inter.is_empty and inter.geom_type not in {'GeometryCollection'} else np.array(nearest_points(rp,source)[0].coords[0])
   a=np.deg2rad(pca['axis_deg']);u=np.array([np.cos(a),np.sin(a)]);proj=vertices(rp)@u;lo,hi=min(proj),max(proj);frac=float((q@u-lo)/max(hi-lo,1e-6))
   if gap<=12 and (frac<=.25 or frac>=.75):
    direction=u if (r-q)@u>0 else -u
    add('explicit_road_away_from_named_junction',angle(direction),'road_travel','explicit','coming from named road; source junction in outer road quartile; directed PCA tangent points into reference road',refs+h['resolved_ids'],.88,meta={'axis_ratio':pca['eigenvalue_ratio'],'source_gap_m':float(gap),'junction_axis_fraction':frac,'junction_xy':q.tolist(),'geometry_is_centerline_proxy':True})
   else:issues.append('road_from_junction_does_not_determine_unique_positive_direction')
 # Road visual-axis alignment has TWO signs unless another named-only order fixes one.
 if pca and cat=='TrafficRoad' and re.search(r'\b(?:viewing|viewed|with)\b.{0,100}\bvertic',text,re.I):
  for delta in [0,180]:add('view_road_vertical',pca['axis_deg']+delta,'image_layout','candidate','road shown vertical; positive image-up sign unspecified',refs,.5)
 if pca and cat=='TrafficRoad' and re.search(r'\b(?:viewing|viewed|with)\b.{0,100}\bhorizont',text,re.I):
  for delta in [90,270]:add('view_road_horizontal',pca['axis_deg']+delta,'image_layout','candidate','road shown horizontal; left/right sign unspecified',refs,.5)
 # An independently named B REL A can constrain the shared view without subject GT.
 for match in TOKEN.finditer(text):
  w=match[0].lower()
  if w not in {'left','right','front','behind','above','below'} or match.start()==inp['direction_span'][0]:continue
  rel=PATTERNS[w].match(text[match.start():])
  if not rel:continue
  rh=immediate_name(inp,match.start()+rel.end())
  lhs=[h for h in inp['mentioned'] if h['end']<match.start() and h['resolved_ids'] and re.fullmatch(r'\s*(?:(?:building|road|is|lies|stands|sits|directly|to|the|on|at|in)\s*)*',text[h['end']:match.start()],re.I)]
  if rh is None or len(lhs)!=1:continue
  a=anchor(inp,rh['resolved_ids']);b=anchor(inp,lhs[0]['resolved_ids'])
  if a is None or b is None or np.linalg.norm(b-a)<=1:continue
  off={'left':90,'right':270,'front':0,'behind':180,'above':0,'below':180}[w]
  add('named_pair_shared_frame_constraint',angle(b-a)-off,'image_layout','candidate','independent named-to-named directional statement; common-description-frame assumption still unproven',rh['resolved_ids']+lhs[0]['resolved_ids'],.65)
 # Fixed prior hierarchy, never GT-based tie breaking. Keep conflicting explicit frames.
 explicit=[c for c in out if c['level']=='explicit']
 prioritized=[c for c in explicit if c['family']==language_family(inp)] or explicit
 strict=[]
 if prioritized:
  h0=prioritized[0]['heading_deg']
  if all(adiff(c['heading_deg'],h0)<=15 for c in prioritized):strict=[prioritized[0]['candidate_id']]
 supported=any(c['family']==language_family(inp) for c in out) and language_family(inp)!='unknown'
 state='explicit_computable' if strict else 'candidate_multiple' if supported else 'unknown'
 return {'occurrence_id':inp['occurrence_id'],'family_silver':language_family(inp),'candidates':out,'strict_state':state,'strict_ids':strict,'unknown_heading':not bool(strict),'reasons':issues,'reference_category':cat,'reference_shape_axis':pca,'reference_xy':r.tolist()}

def routing_ids(pred,family):
 # Silver route label only determines a family. It never chooses a best direction.
 pool=[c for c in pred['candidates'] if c['family']==family]
 if family=='unknown':return []
 explicit=[c for c in pool if c['level']=='explicit']
 return [c['candidate_id'] for c in (explicit or pool)]

def run(out):
 X=readlines(out/'public_inputs.jsonl.gz');pred=[generate(x) for x in X];writelines(out/'inference_predictions.jsonl.gz',pred)
 import pandas as pd
 pd.DataFrame([{'occurrence_id':p['occurrence_id'],'family_silver':p['family_silver'],'strict_state':p['strict_state'],'unknown_heading':p['unknown_heading'],'n_candidates':len(p['candidates']),'strict_ids':p['strict_ids'],'reasons':p.get('reasons',[])} for p in pred]).to_csv(out/'routing_rules.csv',index=False)
 writej(out/'inference_manifest.json',{'public_input_sha256':sha(out/'public_inputs.jsonl.gz'),'prediction_sha256':sha(out/'inference_predictions.jsonl.gz'),'GT_opened_by_inference':False,'future_trajectory_opened':False,'route_labels':'text-only silver labels; not verified author-frame labels','label_counts':dict(collections.Counter(p['family_silver'] for p in pred)),'state_counts':dict(collections.Counter(p['strict_state'] for p in pred)),'strict_selection':'language/geometry prior hierarchy; conflicting explicit directions remain unknown; no per-target oracle'})
 print(readj(out/'inference_manifest.json'),flush=True)
if __name__=='__main__':run(BASE/'results')
