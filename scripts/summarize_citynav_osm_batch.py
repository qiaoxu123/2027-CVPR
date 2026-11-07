"""Install immutable batch proposals and export ONLY independently validated scenes.
Road/junction and public-Polygon checks are proxies, never promoted to control GT.
"""
import argparse,json,csv,shutil,hashlib
from pathlib import Path
import numpy as np,rasterio
from scipy.spatial import cKDTree
from shapely.geometry import Polygon,LineString,mapping,box
from tools.osm_map.auto_registration import render_preview
from graphnav.checkpointing import atomic_json

def public_polygon_kind(value):
 return 'road' if value in ['Road','TrafficRoad'] else 'building' if value=='Building' else None

def validated_export(result,row,data):
 v=row.get('validation')
 if row.get('grade')!='verified' or not v or v['n']<5 or v['max_m']>5:raise ValueError('Independent scene validation required; structural cost cannot authorize export')
 M=np.array(row['matrix']);inv=np.linalg.inv(M[:,:2]);features=[];geoms={};skipped=[]
 with rasterio.open(Path(data)/'rgbd'/(row['map_name']+'.tif')) as r:b=r.bounds
 boundary=box(b.left,b.bottom,b.right,b.top)
 for f in result['osm_features']:
  xy=(np.array(f['xy'])-M[:,2])@inv.T;closed=len(xy)>3 and np.linalg.norm(xy[0]-xy[-1])<.1
  if f['kind']!='road' and not closed:skipped.append(dict(id=f['id'],reason='unclosed_nonroad_way'));continue
  g=Polygon(xy) if closed else LineString(xy)
  if not g.is_valid or g.is_empty:skipped.append(dict(id=f['id'],reason='invalid_geometry'));continue
  if not g.intersects(boundary):continue
  roles=[f['kind']]
  if f['tags'].get('amenity')=='parking' and 'parking' not in roles:roles.append('parking')
  features.append(dict(id=f['id'],kind=f['kind'],roles=roles,tags=f['tags'],osm_version=f.get('osm_version'),osm_timestamp=f.get('osm_timestamp'),geometry=mapping(g),fully_in_scene=boundary.contains(g)));geoms[f['id']]=g
 edges=[];roads={}
 for f in features:
  if f['kind']=='road':
   xy=list(geoms[f['id']].coords) if geoms[f['id']].geom_type=='LineString' else list(geoms[f['id']].exterior.coords)
   for a,b in zip(xy[:-1],xy[1:]):
    ka=','.join(f'{z:.1f}' for z in a);kb=','.join(f'{z:.1f}' for z in b);roads.setdefault(ka,set()).add(kb);roads.setdefault(kb,set()).add(ka)
 for i,f in enumerate(features):
  a=geoms[f['id']]
  for h in features[i+1:]:
   b=geoms[h['id']];d=a.distance(b)
   if a.contains(b):edges.append(dict(subject=f['id'],relation='contains',object=h['id']))
   elif b.contains(a):edges.append(dict(subject=h['id'],relation='contains',object=f['id']))
   elif a.intersects(b):edges.append(dict(subject=f['id'],relation='intersects',object=h['id']))
   if d<=10:edges.append(dict(subject=f['id'],relation='near_10m',object=h['id'],distance_m=d))
 return dict(map_name=row['map_name'],local_xy_features=features,relations=edges,road_graph={k:sorted(v) for k,v in roads.items()},independent_validation=v,registration_matrix=row['matrix'],source_osm=result['osm'],skipped=skipped,scope='research public geometry export; not connected to model; full intersecting geometry retained',limitations='OSM relation/multipolygon holes not reconstructed; node-only POIs not included; near_10m is a graph definition, not a learned language relation')

def road_check(result,matrix):
 junctions=result.get('road_junction_proxy_xy',[]);parts=[]
 for f in result['osm_features']:
  if f['kind']=='road' and f['tags'].get('highway') not in ['footway','path','steps','cycleway','pedestrian']:
   p=np.array(f['xy'])
   for a,b in zip(p[:-1],p[1:]):parts.append(a+(b-a)*np.linspace(0,1,max(2,int(np.linalg.norm(b-a))+1))[:,None])
 if not parts or not junctions:return dict(n=0,reason='No major roads or road-junction proxies')
 src=np.array(junctions);M=np.array(matrix);en=src@M[:,:2].T+M[:,2];d=cKDTree(np.concatenate(parts)).query(en)[0]
 return dict(n=len(d),median_nearest_road_m=float(np.median(d)),fraction_near_5m=float(np.mean(d<=5)),note='Roads were not optimized, but RGB skeleton junctions have unknown identity/semantic quality. NOT true geographic validation.')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--store',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args();summary=json.load(open(a.run/'summary.json'));a.report.mkdir(parents=True,exist_ok=True);exportdir=a.store/'validated_osm_exports';exportdir.mkdir(exist_ok=True);proposals=a.store/'auto_proposals';proposals.mkdir(exist_ok=True);objects=json.load(open(a.data/'cityrefer_original/objects.json'));coverage=[]
 for row in summary['rows']:
  name=row['map_name'];src=a.run/name;dest=proposals/name;dest.mkdir(exist_ok=True)
  if row['status']!='complete':continue
  r=json.load(open(src/'result.json'));local_matrix=r['matrix'];r.update(matrix=row['matrix'],scene_local_candidate_matrix=local_matrix,selected=row['selected'],grade=row['grade'],validation=row['validation'],verified=row['grade']=='verified',structure_cost_selected=row['structure_cost_local'] if row['selected']=='scene_similarity' else row['structure_cost_initial'],initialization_source='frozen ec492c11 city seed, not current-scene manual fit')
  r['per_map_validation']={name:row['validation']} if row.get('validation') else {}
  controls=json.load(open(a.run/(row['city']+'_controls_snapshot.json')));r['controls_fingerprint']=hashlib.sha256(json.dumps(controls['points'],sort_keys=True).encode()).hexdigest();row['road_structural_check']=road_check(r,row['matrix']);r['road_structural_check']=row['road_structural_check'];r['status']='proposal'
  # Existing accepted transforms remain in the private control record; archive old proposal.
  if (dest/'result.json').exists():shutil.copy2(dest/'result.json',dest/'pre_batch_result.json')
  atomic_json(r,src/'selected_result.json');atomic_json(r,dest/'result.json');render_preview(a.data,r,dest/'alignment.png');shutil.copy2(src/'building_proxy.png',dest/'building_proxy.png');shutil.copy2(src/'road_proxy.png',dest/'road_proxy.png')
  if row['grade']=='verified':
   exported=validated_export(r,row,a.data);atomic_json(exported,exportdir/(name+'.json'));atomic_json(exported,a.run/(name+'_validated_osm.json'));coverage.append(dict(map_name=name,features=len(exported['local_xy_features']),buildings=sum(f['kind']=='building' for f in exported['local_xy_features']),roads=sum(f['kind']=='road' for f in exported['local_xy_features']),parking=sum('parking' in f['roles'] for f in exported['local_xy_features']),relations=len(exported['relations']),road_nodes=len(exported['road_graph'])))
   # Public named polygons are a cross-check, not independent control truth.
   refs=[]
   for oid,o in objects.get(name,{}).items():
    if not o.get('name') or len(o.get('contour',[]))<3 or public_polygon_kind(o.get('object_type')) is None:continue
    g=Polygon(o['contour'])
    if not g.is_valid:continue
    candidates=[f for f in exported['local_xy_features'] if f['kind']==public_polygon_kind(o.get('object_type'))]
    if not candidates:continue
    from shapely.geometry import shape
    dists=[(g.distance(shape(f['geometry'])),f['id']) for f in candidates];dist,match=min(dists);refs.append(dict(public_name=o['name'],public_id=str(oid),nearest_compatible_osm=match,polygon_distance_m=dist,note='Nearest geometry proxy; not identity or independent accuracy proof'))
   atomic_json(dict(map_name=name,rows=refs),a.run/(name+'_public_polygon_check.json'))
 atomic_json(summary,a.store/'batch34_summary.json');atomic_json(summary,a.report/'summary.json');atomic_json(coverage,a.report/'validated_export_coverage.json')
 with (a.report/'scenes.csv').open('w') as f:
  fields=['map_name','status','grade','selected','independent_n','rmse_m','p95_m','max_m','precise_2m','structure_cost_initial','structure_cost_local','local_max_corner_shift_m'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in summary['rows']:
   v=r.get('validation') or {};w.writerow({**{k:r.get(k) for k in fields if k not in ['independent_n','rmse_m','p95_m','max_m']},'independent_n':v.get('n',0),'rmse_m':v.get('rmse_m'),'p95_m':v.get('p95_m'),'max_m':v.get('max_m')})
 print(json.dumps(dict(n=summary['n'],completed=summary['completed'],verified=summary['verified'],pending=summary['pending'],unreliable=summary['unreliable'],exports=coverage)),flush=True)
