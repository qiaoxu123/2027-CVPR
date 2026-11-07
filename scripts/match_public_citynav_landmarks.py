"""Public-object correspondence and object-group holdout. No CityNav tasks/target GT.
Automatic matches remain pending review; proxy heldout residuals are NOT surveyed GT.
"""
import argparse,json,hashlib,xml.etree.ElementTree as ET,math
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher
import cv2,numpy as np
from shapely.geometry import Polygon,LineString,Point,mapping
from shapely.ops import unary_union,transform as shape_transform
from rasterio.warp import transform
import rasterio

def norm(s):return ' '.join(str(s).lower().replace('’',"'").replace('&','and').replace("'",'').replace('.','').replace('-',' ').split()).removeprefix('the ')
def score_name(a,b):return SequenceMatcher(None,norm(a),norm(b)).ratio()
def projection(g,M):return shape_transform(lambda x,y,z=None:(M[0,0]*np.asarray(x)+M[0,1]*np.asarray(y)+M[0,2],M[1,0]*np.asarray(x)+M[1,1]*np.asarray(y)+M[1,2]),g)
def error(M,rows):
 if not rows:return None
 p=np.array([r['source_anchor_xy'] for r in rows]);q=np.array([r['osm_anchor_en'] for r in rows]);d=np.linalg.norm(p@M[:,:2].T+M[:,2]-q,axis=1)
 return dict(n=len(rows),mean_m=float(d.mean()),rmse_m=float(np.sqrt((d*d).mean())),p95_m=float(np.quantile(d,.95)),max_m=float(d.max()),n_groups=len({r['independence_group'] for r in rows}),scope='automatic matched-object anchor residual, not confirmed geographic GT')

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--batch',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);catalog=json.load(open(a.data/'cityrefer_original/objects.json'));maps=sorted((a.data/'rgbd').glob('*.tif'));allrows=[];cities={};sourcehashes={}
 for city in ['birmingham','cambridge']:
  M0=np.array(json.load(open(a.batch/(city+'_city_seed.json')))['matrix']);nodes={};ways={}
  files=sorted(a.batch.glob(city+'_block_*/osm_cache/*.osm'))
  for f in files:
   sourcehashes[str(f.relative_to(a.batch))]=hashlib.sha256(f.read_bytes()).hexdigest();root=ET.fromstring(f.read_bytes())
   for n in root.findall('node'):
    oid=n.attrib['id'];v=int(n.attrib.get('version',0))
    if oid not in nodes or v>nodes[oid]['version']:nodes[oid]=dict(ll=[float(n.attrib['lon']),float(n.attrib['lat'])],version=v,tags={t.attrib['k']:t.attrib['v'] for t in n.findall('tag')})
   for w in root.findall('way'):
    oid=w.attrib['id'];v=int(w.attrib.get('version',0))
    if oid not in ways or v>ways[oid]['version']:ways[oid]=dict(id=oid,version=v,refs=[n.attrib['ref'] for n in w.findall('nd')],tags={t.attrib['k']:t.attrib['v'] for t in w.findall('tag')})
  ids=list(nodes);ll=np.array([nodes[i]['ll'] for i in ids]);e,n=transform('EPSG:4326','EPSG:27700',ll[:,0],ll[:,1]);positions={i:np.array([x,y]) for i,x,y in zip(ids,e,n)};buildings=[];targets=[];named_geoms=defaultdict(list);roads=defaultdict(lambda:dict(nodes=set(),ways=[]))
  for wid,w in ways.items():
   if any(i not in positions for i in w['refs']):continue
   xy=np.array([positions[i] for i in w['refs']]);tags=w['tags'];closed=len(xy)>3 and w['refs'][0]==w['refs'][-1];g=Polygon(xy) if closed else LineString(xy)
   if not g.is_valid:continue
   w['geom']=g
   for key in ['name','name:en','official_name','alt_name','brand']:
    for name in tags.get(key,'').split(';'):
     if len(norm(name))>=4:named_geoms[norm(name)].append(g)
   if tags.get('building') and closed:
    buildings.append(w)
    if tags.get('name'):targets.append(dict(id='way:'+wid,entity='building:'+wid,name=tags['name'],geom=g,source='OSM named building way',osm_ids=['way:'+wid],version=w['version']))
   if tags.get('highway') and tags.get('name') and tags.get('bridge')!='yes' and tags.get('tunnel')!='yes' and tags.get('layer','0')=='0':
    key=norm(tags['name']);roads[key]['nodes'].update(w['refs']);roads[key]['ways'].append(w)
  for nid,node in nodes.items():
   tags=node['tags'];name=tags.get('name') or tags.get('brand');pt=Point(positions[nid])
   if not name or len(norm(name))<4:continue
   named_geoms[norm(name)].append(pt)
   # A POI label is not itself a building footprint. Keep all containing alternatives.
   containers=[w for w in buildings if w['geom'].contains(pt)]
   for w in containers:targets.append(dict(id='node:'+nid+'@way:'+w['id'],entity='building:'+w['id'],name=name,geom=w['geom'],source='OSM named POI inside building',osm_ids=['node:'+nid,'way:'+w['id']],version=node['version']))
  neighbour_cache={}
  def context(source,others,target):
   closest=[];used=set([norm(source['name'])]);c=source['geom'].centroid
   for o in sorted(others,key=lambda v:v['geom'].distance(c)):
    name=norm(o['name'])
    if name in used:continue
    used.add(name);closest.append(o)
    if len(closest)==5:break
   proof=[]
   for o in closest:
    name=norm(o['name'])
    if name not in neighbour_cache:neighbour_cache[name]=[g for k,gg in named_geoms.items() if score_name(name,k)>=.92 for g in gg]
    distance=min((g.distance(target['geom'].centroid) for g in neighbour_cache[name]),default=float('inf'));proof.append(dict(name=o['name'],supported=distance<=150,distance_m=float(distance) if math.isfinite(distance) else None))
   return sum(p['supported'] for p in proof)/max(len(proof),1),proof
  rows=[]
  for raster in [m for m in maps if m.stem.startswith(city+'_')]:
   m=raster.stem
   with rasterio.open(raster) as r:bb=r.bounds
   public=[]
   for oid,o in catalog.get(m,{}).items():
    if not o.get('name') or len(o.get('contour',[]))<3:continue
    g=Polygon(o['contour'])
    if g.is_valid and g.area>=4:public.append(dict(id=str(oid),name=o['name'],typ=o['object_type'],geom=g))
   for src in [o for o in public if o['typ']=='Building']:
    source_en=projection(src['geom'],M0);sc=np.array(source_en.centroid.coords[0]);candidates=[]
    for t in targets:
     ns=score_name(src['name'],t['name'])
     if ns<.85:continue
     tc=np.array(t['geom'].centroid.coords[0]);dist=np.linalg.norm(sc-tc)
     if dist>100:continue
     moved=shape_transform(lambda x,y,z=None:(np.asarray(x)+tc[0]-sc[0],np.asarray(y)+tc[1]-sc[1]),source_en);area=min(moved.area,t['geom'].area)/max(moved.area,t['geom'].area);shape=moved.intersection(t['geom']).area/max(moved.union(t['geom']).area,1e-6);ctx,proof=context(src,public,t);confidence=.35*ns+.25*shape+.20*ctx+.10*np.exp(-dist/40)+.10*area;candidates.append(dict(target=t,name_score=ns,shape_iou=shape,area_ratio=area,context_score=ctx,context_proof=proof,initial_distance_m=float(dist),confidence=float(confidence)))
    candidates.sort(key=lambda x:(-x['confidence'],x['target']['id']));best=candidates[0] if candidates else None
    if not best:continue
    # Competing POI names on the SAME physical building are not independent alternatives.
    alternatives=[v for v in candidates[1:] if v['target']['entity']!=best['target']['entity']];margin=best['confidence']-(alternatives[0]['confidence'] if alternatives else 0);t=best['target'];near_border=src['geom'].distance(Polygon([(bb.left,bb.bottom),(bb.right,bb.bottom),(bb.right,bb.top),(bb.left,bb.top)]).boundary)<2
    reasons=[]
    for bad,reason in [(best['name_score']<.90,'name_similarity_low'),(best['shape_iou']<.55,'shape_mismatch'),(best['area_ratio']<.5,'area_mismatch'),(sum(v['supported'] for v in best['context_proof'])<2,'insufficient_neighbour_evidence'),(margin<.08,'ambiguous_same_name_instance'),(near_border,'partial_source_polygon'),(best['confidence']<.82,'confidence_below_fixed_threshold')]:
     if bad:reasons.append(reason)
    identity=f"{city}|{m}|{src['id']}|{t['id']}";rid=hashlib.sha256(identity.encode()).hexdigest()[:20];details={k:v for k,v in best.items() if k!='target'};rows.append(dict(match_id=rid,city=city,map_name=m,kind='building',source_id=src['id'],source_name=src['name'],osm_name=t['name'],source_polygon=mapping(src['geom']),osm_polygon_en=mapping(t['geom']),source_anchor_xy=list(src['geom'].centroid.coords[0]),osm_anchor_en=list(t['geom'].centroid.coords[0]),osm_ids=t['osm_ids'],osm_version=t['version'],source=t['source'],target_entity=t['entity'],physical_tokens=[t['entity']],high_confidence=not reasons,status='pending' if not reasons else 'rejected',reasons=reasons,margin=float(margin),**details))
   # Named-road intersection controls require actual shared OSM nodes and two source regions.
   groups=defaultdict(list)
   for o in public:
    if o['typ'] in ['Road','TrafficRoad']:groups[norm(o['name'])].append(o['geom'])
   keys=sorted(k for k in groups if k in roads)
   for i,k in enumerate(keys):
    for h in keys[i+1:]:
     common=roads[k]['nodes']&roads[h]['nodes']
     if not common:continue
     overlap=unary_union(groups[k]).intersection(unary_union(groups[h]))
     if overlap.is_empty or not 1<overlap.area<400:continue
     point=np.array(overlap.centroid.coords[0]);pred=point@M0[:,:2].T+M0[:,2];near=sorted((float(np.linalg.norm(positions[n]-pred)),n) for n in common)
     if near[0][0]>80:continue
     dist,nid=near[0];ambiguous=len(near)>1 and near[1][0]-dist<15;conf=.9+.1*np.exp(-dist/40);rid=hashlib.sha256(f'{m}|{k}|{h}|{nid}'.encode()).hexdigest()[:20]
     rows.append(dict(match_id=rid,city=city,map_name=m,kind='road_junction',source_id=k+' & '+h,source_name=k+' & '+h,osm_name=k+' & '+h,source_polygon=mapping(overlap),osm_polygon_en=mapping(Point(positions[nid]).buffer(3)),source_anchor_xy=point.tolist(),osm_anchor_en=positions[nid].tolist(),osm_ids=['node:'+nid],osm_version=nodes[nid]['version'],source='Two named source road polygons + shared OSM road node',target_entity='junction:'+nid,physical_tokens=['road:'+city+':'+k,'road:'+city+':'+h],name_score=1.,shape_iou=None,context_score=1.,context_proof=['shared named-road OSM topology'],initial_distance_m=dist,confidence=float(conf),margin=None,high_confidence=not ambiguous,status='pending' if not ambiguous else 'rejected',reasons=['ambiguous_road_junction'] if ambiguous else []))
   print(city,m,'matches_so_far',len(rows),flush=True)
  from tools.osm_map.landmark_groups import grouped_fit
  high=[r for r in rows if r['high_confidence']];M,status,fit,val=grouped_fit(rows,M0)
  # Prediction sealed before reading any validation residuals.
  (a.output/(city+'_prediction_frozen.json')).write_text(json.dumps(dict(matrix=M.tolist(),status=status,fit_ids=[r['match_id'] for r in fit],validation_ids=[r['match_id'] for r in val]),indent=2))
  for r in rows:
   pred=np.array(r['source_anchor_xy'])@M[:,:2].T+M[:,2];r['anchor_residual_m']=float(np.linalg.norm(pred-r['osm_anchor_en']));r['confirmed']=False;r.setdefault('split','excluded');r.setdefault('independence_group',None)
  permap={m.stem:dict(auto_heldout_proxy=error(M,[r for r in val if r['map_name']==m.stem]),manual_independent=None,status='pending_identity_review') for m in maps if m.stem.startswith(city+'_')};cities[city]=dict(matrix=M.tolist(),initial_matrix=M0.tolist(),status=status,matches=len(rows),high_confidence_rows=len(high),fit_objects=len(fit),validation_objects=len(val),fit_groups=len({r['independence_group'] for r in fit}),validation_groups=len({r['independence_group'] for r in val}),initial_validation_proxy=error(M0,val),optimized_validation_proxy=error(M,val),per_scene=permap,automatic_confirmed_geographic_scenes=0)
  allrows+=rows
 (a.output/'matches.json').write_text(json.dumps(allrows,indent=2));(a.output/'summary.json').write_text(json.dumps(dict(seed=42,cities=cities,n=len(allrows),new_independently_verified_scenes=0,source_osm_sha256=sourcehashes,public_catalog_sha256=hashlib.sha256((a.data/'cityrefer_original/objects.json').read_bytes()).hexdigest(),confidence_note='Fixed heuristic score, not calibrated correctness probability',no_target_gt=True,no_validation_feedback_in_fit=True,automatic_identity_matches_are_not_pixel_or_geographic_gold=True),indent=2));print('DONE',flush=True)
if __name__=='__main__':main()
