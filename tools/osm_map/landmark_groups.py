"""Object/group isolation for public landmark correspondences; no validation residual inputs."""
from collections import defaultdict
import hashlib,cv2,numpy as np
from shapely.geometry import shape

def grouped_fit(rows,initial):
 for r in rows:r.update(split='excluded',independence_group=None,representative=False,ransac_fit_inlier=None)
 high=[r for r in rows if r['high_confidence']];parent=list(range(len(high)));tokens={}
 def find(x):
  while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
  return x
 def union(i,j):parent[find(i)]=find(j)
 for i,r in enumerate(high):
  for token in r['physical_tokens']:
   if token in tokens:union(i,tokens[token])
   tokens[token]=i
 shapes={i:(shape(r['source_polygon']),shape(r['osm_polygon_en'])) for i,r in enumerate(high) if r['kind']=='building'}
 for i,(g,og) in shapes.items():
  for j,(h,oh) in shapes.items():
   if j>=i:continue
   overlap=g.intersects(h) and g.intersection(h).area/max(min(g.area,h.area),1e-6)>.5
   # Conservative: adjacent/nested OSM parts are one split group, even if some are distinct terraced houses.
   if overlap or og.distance(oh)<=.5:union(i,j)
 components=defaultdict(list)
 for i,r in enumerate(high):components[find(i)].append(r)
 representatives=[]
 for comp in components.values():
  group=hashlib.sha256('|'.join(sorted({t for r in comp for t in r['physical_tokens']})).encode()).hexdigest()[:16];split='validation' if int(hashlib.sha256(('42|'+group).encode()).hexdigest()[:8],16)%10<3 else 'fit';seen=set()
  for r in sorted(comp,key=lambda r:(-r['confidence'],r['match_id'])):
   r['independence_group']=group;r['split']=split
   if r['target_entity'] not in seen:representatives.append(r);seen.add(r['target_entity']);r['representative']=True
 fit=[r for r in representatives if r['split']=='fit'];val=[r for r in representatives if r['split']=='validation'];assert not {r['independence_group'] for r in fit}&{r['independence_group'] for r in val};M=np.array(initial);status='insufficient_high_confidence_fit'
 if len(fit)>=3:
  X=np.array([r['source_anchor_xy'] for r in fit]);Y=np.array([r['osm_anchor_en'] for r in fit]);cv2.setRNGSeed(42);candidate,inside=cv2.estimateAffinePartial2D(X,Y,method=cv2.RANSAC,ransacReprojThreshold=3,maxIters=5000,confidence=.999)
  if candidate is not None:
   s0=np.sqrt(np.linalg.det(M[:,:2]));s1=np.sqrt(np.linalg.det(candidate[:,:2]));angle=abs(np.arctan2(candidate[1,0],candidate[0,0])-np.arctan2(M[1,0],M[0,0]))
   if .9<s1/s0<1.1 and angle<.1:
    M=candidate;status='automatic_object_fit_proposal'
    for r,flag in zip(fit,inside.ravel()):r['ransac_fit_inlier']=bool(flag)
   else:status='rejected_unstable_city_transform'
 return M,status,fit,val
