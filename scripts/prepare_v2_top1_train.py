"""Rebuild TRAIN queries with exactly the blind public binding used in V2 evaluation.
Private target coordinates, identities and split groups are stored separately.
"""
import argparse,json,sys,hashlib
from pathlib import Path
from types import SimpleNamespace
from collections import Counter,defaultdict
import numpy as np
from graphnav.a3_runtime import public_static_record
from graphnav.checkpointing import atomic_json,file_sha256
from scripts.analyze_instruction_landmark_association import PublicNames
from scripts.unified_pgvb_v3_pilot import _load_base


def main():
 p=argparse.ArgumentParser()
 for k in ['data','sbf-source','source-v2','old-train','output']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();sys.path.insert(0,str(a.sbf_source));from sbfnav.dataset import CityNavDataset
 from sbfnav.coordinates import MapTransform
 ds=CityNavDataset(a.data,'train_seen',version='processed_citynav');assert len(ds)==21878
 a.output.mkdir(parents=True,exist_ok=True)
 if (a.output/'manifest.json').exists():raise FileExistsError('Prepared TRAIN exists; preserve it')
 import torch
 source=torch.load(a.source_v2,weights_only=True,map_location='cpu');cfg=SimpleNamespace(**source['config']['geometry_args']);objects=json.load(open(a.data/'cityrefer/objects.json'));catalog=SimpleNamespace(objects=objects);names=PublicNames(objects);transforms={};old=[json.loads(s) for s in a.old_train.read_text().splitlines() if s.strip()];assert len(old)==len(ds)
 public=[];private=[];parent=list(range(len(ds)));keys={};counts=Counter()
 def find(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 def union(i,j):parent[find(i)]=find(j)
 for i,ep in enumerate(ds):
  if ep.map_name not in transforms:transforms[ep.map_name]=MapTransform.from_raster(a.data/'rgbd'/(ep.map_name+'.tif'),canvas_size=224,field_size=28)
  t=transforms[ep.map_name];refs=tuple(dict.fromkeys(n for m in names.match(ep.map_name,ep.instruction) for n in m['public_names']));query=SimpleNamespace(map_name=ep.map_name,instruction=ep.instruction,reference_names=refs,references=());r=public_static_record(query,t,catalog,max_landmarks=cfg.max_landmarks);assert r['target_xy']==[0.,0.];public.append(r)
  bounds=np.array(r['world_bounds']);target=np.array(ep.target_position[:2]);private.append(dict(index=i,episode=list(ep.episode_id),target_world=target.tolist(),target_xy=((target-bounds[:2])/(bounds[2:]-bounds[:2])).tolist(),city=ep.map_name.split('_')[0]))
  obj=objects.get(ep.map_name,{}).get(str(ep.object_id),{});polykey=json.dumps(sorted(tuple(round(float(v),1) for v in pt) for pt in obj.get('contour',[])))
  groupkeys=[('instance',ep.map_name,str(ep.object_id)),('text',ep.map_name.split('_')[0],' '.join(ep.instruction.lower().split()))]
  if obj.get('contour'):groupkeys.append(('public_shape',ep.map_name.split('_')[0],obj.get('object_type'),polykey))
  for key in groupkeys:
   if key in keys:union(i,keys[key])
   keys[key]=i
  before=sorted(x['name'] for x in old[i].get('landmarks',[]));after=sorted(x['name'] for x in r['landmarks']);counts['name_binding_changed']+=before!=after;counts['new_no_anchor']+=not after;counts['old_no_anchor']+=not before;counts[private[-1]['city']]+=1
  if (i+1)%2000==0:print('prepared',i+1,dict(counts),flush=True)
 components=defaultdict(list)
 for i in range(len(ds)):components[find(i)].append(i)
 for ids in components.values():
  cities={private[i]['city'] for i in ids};assert len(cities)==1
  group=hashlib.sha256('|'.join(sorted(json.dumps(private[i]['episode']) for i in ids)).encode()).hexdigest()[:20]
  split='confirmation' if 'birmingham' in cities else ('development' if int(hashlib.sha256(('42|'+group).encode()).hexdigest()[:8],16)%5==0 else 'fit')
  for i in ids:private[i].update(group=group,split=split)
 assert len({x['group'] for x in private if x['split']=='fit'}&{x['group'] for x in private if x['split']!='fit'})==0
 (a.output/'public_train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in public));atomic_json(private,a.output/'labels_private.json');manifest=dict(n=len(ds),seed=42,counts=dict(counts),splits=dict(Counter(x['split'] for x in private)),groups=len(components),public_sha256=file_sha256(a.output/'public_train.jsonl'),labels_sha256=file_sha256(a.output/'labels_private.json'),dataset_sha256=file_sha256(ds.annotation_path),catalog_sha256=file_sha256(a.data/'cityrefer/objects.json'),source_v2_sha256=file_sha256(a.source_v2),no_val_test=True,private_use='TRAIN loss and split grouping only; public input target_xy is dummy zero',prior_checkpoint_caveat='Original/previous V2 already trained on official TRAIN, including these groups. This holdout measures adaptation regressions, not unseen-to-pretraining generalization. Final official Val never used for tuning.')
 atomic_json(manifest,a.output/'manifest.json');print(json.dumps(manifest,indent=2),flush=True)
if __name__=='__main__':main()
