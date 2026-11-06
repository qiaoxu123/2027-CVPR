"""Target-independent polygon centerline extraction and directed road fields.
Zhang-Suen thinning is implemented with NumPy; no image/model training.
"""
from shared import *
from infer import region,anchor
from shapely.geometry import Point,Polygon
from shapely.ops import nearest_points
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra,connected_components
import cv2

def thin(mask):
 a=np.pad(mask.astype(np.uint8),1)
 for iteration in range(200):
  changed=False
  for step in range(2):
   p=[a[:-2,1:-1],a[:-2,2:],a[1:-1,2:],a[2:,2:],a[2:,1:-1],a[2:,:-2],a[1:-1,:-2],a[:-2,:-2]];n=sum(p);trans=sum((p[i]==0)&(p[(i+1)%8]==1) for i in range(8));kill=(a[1:-1,1:-1]==1)&(n>=2)&(n<=6)&(trans==1)
   if step==0:kill&=(p[0]*p[2]*p[4]==0)&(p[2]*p[4]*p[6]==0)
   else:kill&=(p[0]*p[2]*p[6]==0)&(p[0]*p[4]*p[6]==0)
   changed|=bool(kill.any());a[1:-1,1:-1][kill]=0
  if not changed:break
 return a[1:-1,1:-1]

def skeleton(p,res=1.):
 lo=np.floor(np.array(p.bounds[:2])/res)*res-3*res;hi=np.array(p.bounds[2:]);shape=np.ceil((hi-lo)/res).astype(int)+4
 if np.prod(shape)>3000000:return {'status':'road_grid_too_large'}
 mask=np.zeros((shape[1],shape[0]),np.uint8)
 parts=list(p.geoms) if p.geom_type=='MultiPolygon' else [p]
 for q in parts:
  c=np.rint((np.array(q.exterior.coords)-lo)/res).astype(np.int32);cv2.fillPoly(mask,[c],1)
  for ring in q.interiors:cv2.fillPoly(mask,[np.rint((np.array(ring.coords)-lo)/res).astype(np.int32)],0)
 sk=thin(mask);yx=np.argwhere(sk);lookup={tuple(z):i for i,z in enumerate(yx)};adj={i:{} for i in range(len(yx))}
 for i,z in enumerate(yx):
  for dy,dx in [(0,1),(1,0),(1,1),(1,-1)]:
   j=lookup.get((z[0]+dy,z[1]+dx))
   # Avoid diagonal shortcuts through an already connected corner.
   if j is not None:
    if dy and dx and ((z[0]+dy,z[1]) in lookup or (z[0],z[1]+dx) in lookup):continue
    adj[i][j]=adj[j][i]=float(np.hypot(dy,dx)*res)
 xy=yx[:,::-1]*res+lo
 # Remove short terminal spurs, using road width only; never target locations.
 widths=cv2.distanceTransform(mask,cv2.DIST_L2,5);medianwidth=float(np.median(widths[sk.astype(bool)]))*res if len(yx) else 0
 threshold=max(6.,2.5*medianwidth)
 for _ in range(12):
  doomed=set()
  for start in list(adj):
   if len(adj[start])!=1:continue
   path=[start];cur=start;prev=None;length=0.
   while True:
    nxt=[j for j in adj[cur] if j!=prev]
    if len(nxt)!=1:break
    j=nxt[0];length+=adj[cur][j];prev,cur=cur,j
    if len(adj[cur])!=2:break
    path.append(cur)
   if len(adj[cur])>=3 and length<threshold:doomed.update(path)
  if not doomed:break
  for i in doomed:
   for j in list(adj[i]):adj[j].pop(i,None)
   adj.pop(i,None)
 if len(adj)<5:return {'status':'insufficient_centerline'}
 nodes=sorted(adj);index={k:i for i,k in enumerate(nodes)};rows=[];cols=[];vals=[]
 for i in nodes:
  for j,w in adj[i].items():rows.append(index[i]);cols.append(index[j]);vals.append(w)
 graph=csr_matrix((vals,(rows,cols)),shape=(len(nodes),len(nodes)));ncomp,labels=connected_components(graph);counts=np.bincount(labels);component=np.argmax(counts);keep=np.flatnonzero(labels==component);g=graph[keep][:,keep];coords=xy[nodes][keep];degrees=np.diff(g.indptr);ends=np.flatnonzero(degrees==1)
 if len(ends)<2:return {'status':'closed_or_no_endpoints','components':ncomp}
 # Retain the complete pruned graph for ambiguity diagnostics, not just best path.
 return {'status':'ok','xy':coords,'graph':g,'endpoints':ends,'branch_nodes':int(sum(degrees>2)),'components':ncomp,'component_fraction':float(max(counts)/len(nodes)),'resolution_m':res,'prune_length_m':threshold,'median_halfwidth_m':medianwidth}

def directed_field(inp,road_ids,source_ids,resolution=1.,robust=True):
 p=region(inp,road_ids);q=region(inp,source_ids)
 if p is None or q is None:return {'state':'unknown','reason':'invalid_road_or_source_region'}
 sk=skeleton(p,res=resolution)
 if sk['status']!='ok':return {'state':'unknown','reason':sk['status']}
 xy=sk['xy'];g=sk['graph'];ends=sk['endpoints'];inter=p.intersection(q);gap=p.distance(q)
 if gap>12:return {'state':'unknown','reason':'source_road_not_connected','gap_m':float(gap)}
 source_pt=np.array(inter.centroid.coords[0]) if not inter.is_empty else np.array(nearest_points(p,q)[0].coords[0]);seed=int(np.argmin(np.linalg.norm(xy-source_pt,axis=1)));ds=dijkstra(g,indices=seed);diam=float(np.max(ds[np.isfinite(ds)]));nearestend=int(ends[np.argmin(ds[ends])]);startdist=float(ds[nearestend]);at_end=startdist<=max(12.,.15*diam)
 path_unique=bool(len(ends)==2 and sk['branch_nodes']==0 and sk['component_fraction']>=.95 and at_end)
 source_node=nearestend if at_end else seed;d=dijkstra(g,indices=source_node)
 # Smooth tangents over a 6m geodesic window; forward is increasing distance.
 tangents=[]
 for i in range(len(xy)):
  local=dijkstra(g,indices=i,limit=8.);near=np.flatnonzero(np.isfinite(local));forward=near[d[near]>d[i]+1];back=near[d[near]<d[i]-1]
  if len(forward) and len(back):v=np.mean(xy[forward],axis=0)-np.mean(xy[back],axis=0)
  elif len(forward):v=np.mean(xy[forward],axis=0)-xy[i]
  elif len(back):v=xy[i]-np.mean(xy[back],axis=0)
  else:v=np.zeros(2)
  tangents.append(angle(v) if np.linalg.norm(v)>0 else None)
 result={'state':'unique_field' if path_unique else 'multiple','reason':'simple_centerline_entered_from_endpoint' if path_unique else 'branches_or_internal_entry_or_disconnected_geometry','road_ids':road_ids,'source_ids':source_ids,'origin_xy':xy[source_node].tolist(),'source_junction_xy':source_pt.tolist(),'source_to_endpoint_geodesic_m':startdist,'branch_nodes':sk['branch_nodes'],'endpoints':len(ends),'component_fraction':sk['component_fraction'],'gap_m':float(gap),'resolution_m':sk['resolution_m'],'prune_length_m':sk['prune_length_m'],'points_xy':xy.tolist(),'distance_from_source_m':d.tolist(),'heading_deg':tangents,'edges':[[int(i),int(j)] for i,j in zip(*g.nonzero()) if i<j],'GT_used':False,'identifiability':'spatially varying road direction; no single target-specific yaw selected'}

 if path_unique and robust:
  sensitivity=[]
  for rr in [.75,1.25]:
   alt=directed_field(inp,road_ids,source_ids,resolution=rr,robust=False);entry={'resolution_m':rr,'state':alt['state']}
   if alt['state']=='unique_field':
    other=np.array(alt['points_xy']);ix=np.argmin(np.linalg.norm(xy[:,None,:]-other[None,:,:],axis=2),axis=1);diff=[float(adiff(h,alt['heading_deg'][j])) for h,j in zip(tangents,ix) if h is not None and alt['heading_deg'][j] is not None];entry['heading_error_p90_deg']=float(np.quantile(diff,.9)) if diff else 180.
   sensitivity.append(entry)
  result['geometry_sensitivity']=sensitivity
  if any(z['state']!='unique_field' or z.get('heading_error_p90_deg',180)>15 for z in sensitivity):result.update(state='multiple',reason='centerline_topology_or_tangent_not_stable_across_resolutions')
 return result

if __name__=='__main__':
 X=readlines(OUT/'public_inputs.jsonl.gz');cache={};diag=[]
 from infer import immediate_name
 for x in X:
  if not all(x['scene'][i]['category']=='TrafficRoad' for i in x['reference_ids']):continue
  for m in re.finditer(r'coming\s+from\s+(?:the\s+)?',x['instruction'],re.I):
   h=immediate_name(x,m.end())
   if not h:continue
   key=x['map']+':'+','.join(x['reference_ids'])+'<-'+','.join(h['resolved_ids'])
   if key not in cache:cache[key]=directed_field(x,x['reference_ids'],h['resolved_ids'])
   f=cache[key];diag.append({'id':x['occurrence_id'],'key':key,**{k:v for k,v in f.items() if k not in {'points_xy','distance_from_source_m','heading_deg','edges'}}})
 writelines(OUT/'road_fields.jsonl.gz',[{'key':k,**v} for k,v in cache.items()]);writej(OUT/'road_field_diagnostics.json',diag);print([(r['id'],r['state'],r.get('endpoints'),r.get('branch_nodes'),r['reason']) for r in diag])
