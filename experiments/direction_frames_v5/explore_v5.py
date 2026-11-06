"""Pre-GT exploratory geometry smoothing and endpoint topology audit."""
from v5core import *
from shapely.geometry import Point,LineString
import copy

def smooth_field(f,x):
 if f['state']!='unique_field':return {'state':'unknown','reason':'base_field_not_unique'}
 xy=np.array(f['points_xy']);d=np.array(f['distance_from_source_m']);smooth=np.array([xy[np.abs(d-s)<=4].mean(0) for s in d]);smooth[d.argmin()]=xy[d.argmin()];smooth[d.argmax()]=xy[d.argmax()];p=region(x,f['road_ids']);shift=np.linalg.norm(smooth-xy,axis=1)
 if shift.max()>2 or not all(p.buffer(1).covers(Point(*z)) for z in smooth):return {'state':'unknown','reason':'smoothing_displacement_or_road_boundary_gate','max_displacement_m':float(shift.max())}
 h=[]
 for i,s in enumerate(d):
  a=smooth[(d<s-1)&(d>=s-8)];b=smooth[(d>s+1)&(d<=s+8)];v=(b.mean(0) if len(b) else smooth[i])-(a.mean(0) if len(a) else smooth[i]);h.append(angle(v))
 out=copy.deepcopy(f);out.update(points_xy=smooth.tolist(),heading_deg=h,geometry_variant='F_exploratory_fixed_4m_coordinate_smoothing',max_displacement_m=float(shift.max()),GT_used=False);return out

def run():
 X=readlines(RESULT/'public_inputs.jsonl.gz');F=readlines(RESULT/'road_fields.jsonl.gz');variant=[];topo=[]
 for f in F:
  if not f.get('road_ids'):variant.append({'key':f['key'],'state':'unknown','reason':f['reason']});continue
  x=next(x for x in X if x['map']==f['key'].split(':')[0] and all(i in x['scene'] for i in f['road_ids']+f['source_ids']));v=smooth_field(f,x);variant.append({'key':f['key'],**v});a=region(x,f['road_ids']);b=region(x,f['source_ids']);xy=np.array(f['points_xy']);d=np.array(f['distance_from_source_m']);ends=[xy[d.argmin()],xy[d.argmax()]];contacts=[float(s) for pt,s in zip(xy,d) if b.buffer(1).covers(Point(*pt))]
  topo.append({'key':f['key'],'base_state':f['state'],'source_segments':f['source_ids'],'road_ids':f['road_ids'],'endpoint_source_distances_m':[float(b.distance(Point(*p))) for p in ends],'contact_geodesic_range_m':[min(contacts),max(contacts)] if contacts else None,'road_length_m':float(max(d)),'source_gap_m':float(a.distance(b)),'source_intersection_area_m2':float(a.intersection(b).area),'one_endpoint_in_source_buffer_1m':sum(b.buffer(1).covers(Point(*p)) for p in ends)==1,'independent_source_contacts_both_ends':all(b.buffer(1).covers(Point(*p)) for p in ends),'geometry_sensitivity':f.get('geometry_sensitivity',[]),'GT_used':False})
 nullfields=[]
 for f in F:
  if f['state']!='unique_field':continue
  x=next(x for x in X if x['map']==f['key'].split(':')[0] and all(i in x['scene'] for i in f['road_ids']+f['source_ids']));names=collections.defaultdict(list)
  for i,o in x['scene'].items():
   if o['category']=='TrafficRoad' and o['name'] and i not in f['road_ids']+f['source_ids']:names[name_key(o['name'])].append(i)
  choices=sorted(names);wrong=names[choices[0]] if choices else []
  for mode,aa,bb in [('swap_source_road',f['road_ids'],wrong),('wrong_landmark_binding',wrong,f['source_ids'])]:
   z=directed_field(x,aa,bb) if wrong else {'state':'unknown','reason':'no_alternative_named_road'};nullfields.append({'key':f['key'],'control':mode,'alternative_road_ids':aa,'alternative_source_ids':bb,**z,'GT_used_to_select_alternative':False})
 writelines(RESULT/'negative_fields.jsonl.gz',nullfields)
 profiles=[]
 for f in F:
  if f['state']!='unique_field':continue
  x=next(x for x in X if x['map']==f['key'].split(':')[0] and all(i in x['scene'] for i in f['road_ids']+f['source_ids']))
  for res in [.5,.75,1.25,2.]:profiles.append({'base_key':f['key'],'profile_resolution_m':res,**directed_field(x,f['road_ids'],f['source_ids'],resolution=res,robust=False)})
 writelines(RESULT/'resolution_fields.jsonl.gz',profiles)
 writelines(RESULT/'exploratory_fields.jsonl.gz',variant);savecsv('source_topology_audit',topo);manifest=readj(RESULT/'phase1_manifest.json');manifest.update(exploratory_fields_sha256=sha(RESULT/'exploratory_fields.jsonl.gz'),exploration_code_sha256=sha(HERE/'explore_v5.py'),negative_fields_sha256=sha(RESULT/'negative_fields.jsonl.gz'),resolution_fields_sha256=sha(RESULT/'resolution_fields.jsonl.gz'),phase1_fully_frozen=True);writej(RESULT/'phase1_manifest.json',manifest);print('Exploratory fields frozen',len(variant))
if __name__=='__main__':run()
