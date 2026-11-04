"""Read-only TRAIN/cache audit; original CityNav destination GT is never used."""
import argparse,json,time,sys
from collections import Counter
from pathlib import Path
import numpy as np,torch,shapely
from graphnav.checkpointing import file_sha256,atomic_json
from graphnav.landmark_alias import canonical_name
from graphnav.landmark_region_supervision import public_type,named_union
from graphnav.memory import ObservedRGBMemory

def main():
 p=argparse.ArgumentParser()
 for k in ['annotations','catalog','cache','output','sbf-source','smoke']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();sys.path.insert(0,str(a.sbf_source));from sbfnav.mapdata import GROUND_LEVEL
 raw=json.loads(a.annotations.read_text());cat=json.loads(a.catalog.read_text());names={}
 generic={'building','road','ground','parking','office','offices','hall','chapel','church','library','house','college'}
 for mp,objs in cat.items():
  grouped={}
  for oid,obj in objs.items():
   n=canonical_name(obj.get('name',''))
   if len(n)>=3 and n not in generic:grouped.setdefault(n,[]).append(obj)
  names[mp]=[]
  for n,items in grouped.items():
   typ=public_type([x['object_type'] for x in items]) or 'Other'
   try:geom=named_union([x['contour'] for x in items])
   except ValueError:geom=None
   names[mp].append((n,typ,geom))
 bound=0;type_counts=Counter();queries=0;intersections=0;possible_frames=set();invalid_pose=0;named_rows=[]
 for i,row in enumerate(raw):
  mp=f"{row['area']}_block_{row['block']}";text=' '+canonical_name(row['descriptions'][0])+' ';matches=[x for x in names[mp] if ' '+x[0]+' ' in text]
  if not matches:continue
  bound+=1;queries+=len(matches);type_counts.update(x[1] for x in matches);pose=row['trajectory'][0];height=pose[2]-GROUND_LEVEL[mp];yaw=float(np.arctan2(pose[4],pose[3]));hit=0
  if height<=0:invalid_pose+=1
  else:
   view=shapely.Polygon(ObservedRGBMemory.view_corners(pose[:2],yaw,height))
   for n,typ,geom in matches:
    if typ in ['Building','Road'] and geom is not None and geom.intersection(view).area>1e-3:intersections+=1;hit+=1
   if hit:possible_frames.add((mp,tuple(np.round([*pose[:3],yaw],7))))
  named_rows.append(dict(episode=i,map_name=mp,named_queries=len(matches),geometric_in_view_queries=hit))
 manifest=json.loads((a.cache/'manifest.json').read_text());assert file_sha256(a.cache/'cache.pt')==manifest['cache_sha256'];t=time.perf_counter();data=torch.load(a.cache/'cache.pt',map_location='cpu',weights_only=True);load_s=time.perf_counter()-t;source=[]
 for path,expected in manifest['source_feature_manifests'].items():
  assert file_sha256(path)==expected;value=json.loads(Path(path).read_text());assert value['split']=='train_seen';source.append(dict(path=path,sha256=expected,split=value['split'],frames=value.get('frames'),seconds=value.get('seconds'),yaw=value.get('camera_yaw_rad')))
 counts={};groups={};shas={};maps={}
 for subset in ['fit','dev']:
  rs=[r for r in data['records'] if r['subset']==subset];fs={r['frame_id'] for r in rs};groups[subset]={(data['frames'][r['frame_id']]['city'],canonical_name(data['queries'][r['query_key']]['name'])) for r in rs};shas[subset]={data['frames'][i]['frame']['image_sha256'] for i in fs};maps[subset]={data['frames'][i]['map_name'] for i in fs};counts[subset]=dict(queries=len(rs),frames=len(fs),types=dict(Counter(r['type'] for r in rs)),city_qualified_names=len(groups[subset]),query_keys=len({r['query_key'] for r in rs}),positive_only=sum(not bool((r['labels']==0).any()) for r in rs))
 assert not groups['fit']&groups['dev'] and not shas['fit']&shas['dev'] and not maps['fit']&maps['dev'];assert all((r['labels']==1).any() for r in data['records']);assert all(data['frames'][r['frame_id']]['subset']==r['subset'] for r in data['records'])
 missing=[f['image_path'] for f in data['frames'] if not Path(f['image_path']).is_file()];assert not missing
 smokehash={file_sha256(p) for p in a.smoke.rglob('*.png')};cachehash={f['frame']['image_sha256'] for f in data['frames']};overlap=smokehash&cachehash;assert not overlap
 frame_keys=Counter('named_public_pose' if f['frame']['frame_key'].startswith('named_train') else 'other_train_cache' for f in data['frames'])
 report=dict(TRAIN_instructions=len(raw),conservative_public_name_bound_instructions=bound,public_name_bindings=queries,binding_types=dict(type_counts),binding_policy='Exact canonical public names, non-generic only; overlap allowed, not human parsing accuracy',geometric_in_view_referenced_queries_at_original_start=intersections,unique_original_start_poses_with_geometric_support=len(possible_frames),binding_start_poses_nonpositive_altitude=invalid_pose,geometric_support_caveat='No full-TRAIN RGB/depth validity/weak-mask scan yet; this is geometric inventory, not confirmed visibility or full supervision count',cached_counts=counts,unique_cached_frames=len(data['frames']),cache_supervised_queries=len(data['records']),cache_unique_query_keys=len(data['queries']),cache_feature_coverage=1.,cache_missing_feature_queries=0,cache_missing_rgb_files=missing,cache_load_seconds=load_s,source_feature_manifests=source,frame_provenance=dict(frame_keys),cache_no_eligible_named_region_frames=manifest['excluded'].get('no_eligible_named_region'),cache_insufficient_valid_patches_frames=manifest['excluded'].get('insufficient_valid_patches'),train_dev_city_map_name_RGB_disjoint=True,ValSeen64_used_for_training=False,ValSeen_smoke_RGB_hash_overlap=len(overlap),destination_GT_used=False,teacher_trajectory_as_model_input=False,observation_policy='Reused verified TRAIN legal observed-view features; public named-pose augmentation and earlier TRAIN observations, yaw=0. Original runtime yaw unchanged. No future observation/history supplied.',scope='All 6976 cached queries are real weak-supervision pool, not all 22002 instruction/pose pairs. No dataset-size reduction for speed test.',cache_sha256=manifest['cache_sha256'],source_sha256={str(p):file_sha256(p) for p in [a.annotations,a.catalog,a.cache/'manifest.json',Path(__file__)]})
 atomic_json(report,a.output);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
