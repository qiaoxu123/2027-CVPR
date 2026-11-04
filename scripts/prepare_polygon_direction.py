"""Build public full-Polygon inputs and grouped TRAIN partitions without GT XY.

Official object IDs are used solely for split isolation, never prediction.
Coordinates of CityNav destination labels are exported later by a separate tool.
"""
import argparse,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np,rasterio,shapely
from shapely.ops import unary_union
from graphnav.landmark_alias import canonical_name
from graphnav.local_landmark_anchor import named_public_polygons
from graphnav.geometry import grid_centers

class Groups:
 def __init__(self):self.parent={}
 def find(self,k):
  self.parent.setdefault(k,k)
  if self.parent[k]!=k:self.parent[k]=self.find(self.parent[k])
  return self.parent[k]
 def join(self,a,b):self.parent[self.find(a)]=self.find(b)

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--catalog',type=Path,required=True);p.add_argument('--annotations',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--old-public',type=Path,required=True);a=p.parse_args();o=a.directory;assert not (o/'public.json').exists();selected=json.loads((o/'selected_text.json').read_text());cat=json.loads(a.catalog.read_text());raw=json.loads(a.annotations.read_text());assert len(raw)==22002
 for r in selected:assert raw[r['episode']]['descriptions'][0]==r['instruction']
 # Only identity metadata below, never target_positions or any destination XY.
 keys={};old_keys=set();old=json.loads(a.old_public.read_text())
 def identity(r):
  episode=r['episode'];mp=r['map_name'];name=canonical_name(r['landmark_name']);city=mp.split('_block_')[0]
  return [('episode',str(episode)),('landmark',city,name)]+[('instance',mp,str(k)) for k in raw[episode]['object_ids']]
 for r in old:old_keys.update(identity(r))
 g=Groups()
 for r in selected:
  kk=identity(r);keys[r['candidate_id']]=kk
  for k in kk[1:]:g.join(kk[0],k)
 grouped=defaultdict(list)
 for r in selected:grouped[g.find(keys[r['candidate_id']][0])].append(r)
 groupinfo=[]
 for rr in grouped.values():
  groupkeys=set(k for r in rr for k in keys[r['candidate_id']]);gid=hashlib.sha256(json.dumps(sorted(groupkeys)).encode()).hexdigest();exposed=bool(groupkeys&old_keys);groupinfo.append(dict(group=gid,previous_exposure=exposed,candidate_ids=[r['candidate_id'] for r in rr],public_landmarks=sorted({r['map_name'].split('_block_')[0]+':'+r['landmark_name'] for r in rr}),words=dict(Counter(r['word'] for r in rr))))
 fresh=sorted([g for g in groupinfo if not g['previous_exposure']],key=lambda g:g['group']);n=max(1,len(fresh)//3) if len(fresh)>=3 else 0;assignment={g['group']:('confirm' if i<n else 'dev' if i<2*n else 'fit') for i,g in enumerate(fresh)};membership={}
 for g in groupinfo:
  g['subset']=assignment.get(g['group'],'fit')
  for k in g['candidate_ids']:membership[k]=(g['subset'],g['group'])
 public=[];reject=[];cache={}
 for r in selected:
  mp=r['map_name'];bound=named_public_polygons(cat[mp],r['landmark_name'])
  if set(bound['ids'])!=set(r['public_object_ids']):reject.append(dict(candidate_id=r['candidate_id'],reason='invalid public fragment: reject whole relation, no GT-based fragment selection',invalid_ids=bound['invalid_ids']));continue
  if mp not in cache:
   with rasterio.open(a.data_root/'rgbd'/f'{mp}.tif') as t:cache[mp]=np.array([t.bounds.left,t.bounds.bottom,t.bounds.right,t.bounds.top])
  b=cache[mp];region=unary_union([shapely.Polygon(x) for x in bound['contours']]).intersection(shapely.box(*b));span=b[2:]-b[:2];world=b[:2]+grid_centers(28).numpy()*span;dx,dy=span/28;cells=shapely.box(world[:,0]-dx/2,world[:,1]-dy/2,world[:,0]+dx/2,world[:,1]+dy/2);mass=shapely.area(shapely.intersection(cells,region))
  if mass.sum()<=0:reject.append(dict(candidate_id=r['candidate_id'],reason='public Polygon has no map area'));continue
  subset,gid=membership[r['candidate_id']];public.append(dict(index=len(public),candidate_id=r['candidate_id'],episode=r['episode'],map_name=mp,subset=subset,group=gid,word=r['word'],direction_id=['north','south','east','west'].index(r['word']),landmark_name=r['landmark_name'],public_object_ids=bound['ids'],public_types=r['public_types'],public_polygons=bound['contours'],anchor=(mass/mass.sum()).astype(np.float32).reshape(28,28).tolist(),bounds=b.tolist(),span=span.tolist(),instruction=r['instruction'],text_source_span=r['relation_span'],target_visibility='unknown'))
 # Independence is asserted by groups AND by official identities, not rows.
 splitkeys={s:set(k for r in selected if membership[r['candidate_id']][0]==s for k in keys[r['candidate_id']]) for s in ['fit','dev','confirm']}
 assert not splitkeys['fit']&splitkeys['dev'] and not splitkeys['fit']&splitkeys['confirm'] and not splitkeys['dev']&splitkeys['confirm'];assert not splitkeys['confirm']&old_keys and not splitkeys['dev']&old_keys
 for name,x in [('public.json',public),('groups.json',groupinfo),('geometry_rejections.json',reject)]: (o/name).write_text(json.dumps(x,ensure_ascii=False,indent=2))
 manifest=dict(total_pairs=len(public),groups=len(groupinfo),fresh_groups=len(fresh),previously_exposed_groups=sum(g['previous_exposure'] for g in groupinfo),counts={s:dict(Counter(r['word'] for r in public if r['subset']==s)) for s in ['fit','dev','confirm']},maps={s:sorted({r['map_name'] for r in public if r['subset']==s}) for s in ['fit','dev','confirm']},group_split='Connected components of city-qualified public landmark names, Episode and map-qualified official target IDs. New groups SHA256 sorted; first floor(n/3) confirm, next floor(n/3) dev; rest fit. Previously exposed components forced fit.',independence='No landmark/Episode/official-instance identity overlap; dev/confirm exclude all previous154 pair exposure. Maps may overlap and are reported.',target_coordinates_used=False,annotations_used_fields=['descriptions','area','block','object_ids (split metadata only)'],source_sha256={str(p):digest(p) for p in [o/'selected_text.json',o/'public.json',o/'groups.json',a.annotations,Path(__file__)]})
 (o/'split_seal.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({k:v for k,v in manifest.items() if k!='source_sha256'},indent=2))
if __name__=='__main__':main()
