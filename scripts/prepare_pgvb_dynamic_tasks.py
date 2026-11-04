"""Seal map-disjoint TRAIN public-name tasks using real starts, never target GT."""
import argparse,collections,gzip,hashlib,json,sys
from pathlib import Path
import numpy as np,torch
from graphnav.checkpointing import file_sha256
from graphnav.landmark_alias import canonical_name
from graphnav.landmark_navigation_diagnostic import public_region
from graphnav.landmark_region_supervision import public_type
from scripts.analyze_instruction_landmark_association import iter_train


def main():
 p=argparse.ArgumentParser()
 for k in ['train','structured','catalog','old-tasks','pgvb-data','data-root','sbf-source','output']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();assert not a.output.exists();sys.path.insert(0,str(a.sbf_source))
 from sbfnav.dataset import _pose4
 from sbfnav.coordinates import MapTransform
 from sbfnav.mapdata import GROUND_LEVEL
 catalog=json.loads(a.catalog.read_text());old={(t['map_name'],canonical_name(t['landmark_name'])) for t in json.loads(a.old_tasks.read_text())};d=torch.load(a.pgvb_data,weights_only=True,map_location='cpu');used={tuple(k.split('::',1)) for k in d['queries']};del d
 with gzip.open(a.structured,'rt') as f:parsed=json.load(f)
 names={}
 for x in parsed:
  for lm in x['landmarks']:
   for name in lm['public_names']:
    key=(x['map_name'],canonical_name(name))
    if key not in old|used:names[key]=name
 definitions={};transforms={};excluded=collections.Counter()
 for key,name in names.items():
  mp=key[0];group=[(oid,o) for oid,o in catalog[mp].items() if canonical_name(o.get('name',''))==key[1]];typ=public_type([o['object_type'] for _,o in group])
  if typ not in ['Building','Road']:excluded['unsupported_type']+=1;continue
  if mp not in transforms:
   t=MapTransform.from_raster(a.data_root/'rgbd'/f'{mp}.tif');transforms[mp]=[t.x_min,t.y_min,t.x_max,t.y_max]
  try:region=public_region([o['contour'] for _,o in group],transforms[mp])
  except ValueError:excluded['invalid_public_geometry']+=1;continue
  definitions[key]=dict(region=region,name=name,group=group,type=typ)
 by_index={x['episode']:x for x in parsed};candidates={}
 for i,row in iter_train(a.train):
  x=by_index[i];mp=f"{row['area']}_block_{row['block']}";assert mp==x['map_name'] and row['descriptions'][0]==x['original_text']
  pose=_pose4(row['trajectory'][0]);height=pose[2]-GROUND_LEVEL[mp]
  if not 10<=height<=100:excluded['illegal_start_height']+=1;continue
  for lm in x['landmarks']:
   for name in lm['public_names']:
    key=(mp,canonical_name(name))
    if key not in definitions:continue
    q=definitions[key];distance=float(q['region'].distance(__import__('shapely').Point(pose[:2])))
    if distance<=20:excluded['already_within_public_region20']+=1;continue
    priority=hashlib.sha256(f'pgvb-dynamic-v1:{i}:{key}'.encode()).hexdigest()
    if key not in candidates or priority<candidates[key][0]:
     candidates[key]=(priority,dict(task_id=mp+'::'+key[1],map_name=mp,source_train_index=i,landmark_name=q['name'],category=q['type'],instruction=f'Navigate to {q["name"]}.',start_pose=pose.tolist(),world_bounds=transforms[mp],public_polygons=[o['contour'] for _,o in q['group']],public_object_ids=[oid for oid,_ in q['group']],public_types=sorted({o['object_type'] for _,o in q['group']})))
 maps=sorted({key[0] for key in candidates},key=lambda mp:hashlib.sha256(('split:'+mp).encode()).hexdigest());assignment={mp:'develop' if j<len(maps)//3 else 'confirm' for j,mp in enumerate(maps)}
 a.output.mkdir(parents=True);proof={}
 for subset,quota in [('develop',12),('confirm',24)]:
  chosen=[]
  for typ in ['Building','Road']:
   permap=collections.defaultdict(list)
   for _,t in candidates.values():
    if assignment[t['map_name']]==subset and t['category']==typ:permap[t['map_name']].append(t)
   for mp in permap:permap[mp].sort(key=lambda t:hashlib.sha256(t['task_id'].encode()).hexdigest())
   selected=[]
   while len(selected)<quota and any(permap.values()):
    for mp in sorted(permap):
     if permap[mp] and len(selected)<quota:selected.append(permap[mp].pop(0))
   if len(selected)!=quota:raise ValueError(f'{subset}/{typ} only {len(selected)}/{quota}, pool {collections.Counter(t["type"] for t in definitions.values())}')
   chosen.extend(selected)
  chosen.sort(key=lambda t:t['task_id'])
  for i,t in enumerate(chosen):t['index']=i
  folder=a.output/subset;folder.mkdir();(folder/'tasks_public.json').write_text(json.dumps(chosen,indent=2));manifest=dict(tasks=len(chosen),source_split='TRAIN',subset=subset,categories=dict(collections.Counter(t['category'] for t in chosen)),maps=sorted({t['map_name'] for t in chosen}),tasks_sha256=file_sha256(folder/'tasks_public.json'),prior64_and_PGVB_cache_names_excluded=True,selection='Deterministic name/map balanced; real official TRAIN start only; no target GT or outcomes; public-region initial outside20m')
  (folder/'manifest.json').write_text(json.dumps(manifest,indent=2));proof[subset]=manifest
 proof['map_assignment']=assignment;proof['input_sha256']={str(f):file_sha256(f) for f in [a.train,a.structured,a.catalog,a.old_tasks,a.pgvb_data,Path(__file__)]};proof['excluded']=dict(excluded);proof['candidate_groups']=len(candidates);(a.output/'selection_proof.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof,indent=2))
if __name__=='__main__':main()
