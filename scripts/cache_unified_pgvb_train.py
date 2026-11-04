"""Incremental full-TRAIN t=0 landmark cache, no destination GT/future frames.

Only publicly named Building/Road queries with reliable current-view positive
patches are retained. RGB and depth-validity keys prevent repeated vision work.
Chunk progress survives interruption; historical caches are read-only.
"""
import argparse,json,time,sys,hashlib,gc
from pathlib import Path
from collections import defaultdict,Counter
import cv2,numpy as np,torch,shapely
from graphnav.checkpointing import atomic_json,atomic_save,file_sha256
from graphnav.landmark_alias import canonical_name
from graphnav.landmark_region_supervision import named_union,public_type
from graphnav.local_landmark_anchor import local_landmark_anchor,local_anchor_tokens
from graphnav.renderer import CityNavObservedViewRenderer
from graphnav.pgvb_observation import encode_landmark_observation
from graphnav.appearance_matching import ObservedSigLIPMatchingEncoder
from graphnav.geometry import grid_centers

GENERIC={'building','road','ground','parking','office','offices','hall','chapel','church','library','house','college'}
def digest_bytes(x):return hashlib.sha256(x).hexdigest()
def main():
 p=argparse.ArgumentParser()
 for k in ['annotations','catalog','data-root','sbf-source','old-cache','output']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--resume',action='store_true');a=p.parse_args();torch.set_num_threads(4);sys.path.insert(0,str(a.sbf_source));from sbfnav.mapdata import GROUND_LEVEL
 start=time.perf_counter();raw=json.loads(a.annotations.read_text());assert len(raw)==22002;catalog=json.loads(a.catalog.read_text());named={};inventory={};excluded=Counter();metrics=[]
 for mp,objects in catalog.items():
  grouped=defaultdict(list)
  for oid,obj in objects.items():
   n=canonical_name(obj.get('name',''))
   if len(n)>=3 and n not in GENERIC:grouped[n].append(obj)
  named[mp]={}
  for n,items in grouped.items():
   typ=public_type([q['object_type'] for q in items])
   if typ not in ['Building','Road']:continue
   try:region=named_union([q['contour'] for q in items])
   except ValueError:continue
   named[mp][n]=dict(key=mp+'::'+n,name=items[0]['name'],type=typ,contours=[q['contour'] for q in items],region=region)
 for i,row in enumerate(raw):
  mp=f"{row['area']}_block_{row['block']}";text=' '+canonical_name(row['descriptions'][0])+' ';matches=[n for n in named[mp] if ' '+n+' ' in text]
  if not matches:continue
  pose=row['trajectory'][0];xyzyaw=[float(x) for x in pose[:3]]+[float(np.arctan2(pose[4],pose[3]))]
  if pose[2]<=GROUND_LEVEL[mp]:excluded['nonpositive_original_start_altitude']+=1;continue
  # Public geometry only: avoid even rendering provably disjoint names.
  from graphnav.memory import ObservedRGBMemory
  view=shapely.Polygon(ObservedRGBMemory.view_corners(xyzyaw[:2],xyzyaw[3],xyzyaw[2]-GROUND_LEVEL[mp]));matches=[n for n in matches if named[mp][n]['region'].intersection(view).area>1e-3]
  if not matches:continue
  key=(mp,tuple(np.round(xyzyaw,7)));entry=inventory.setdefault(key,dict(map_name=mp,pose=xyzyaw,names=set(),source_episodes=[]));entry['names'].update(matches);entry['source_episodes'].append(i)
 wanted=sorted(inventory.values(),key=lambda r:(r['map_name'],r['pose']));fingerprint=dict(annotations=file_sha256(a.annotations),catalog=file_sha256(a.catalog),code=file_sha256(__file__),old_cache_manifest=file_sha256(a.old_cache/'manifest.json'),wanted=len(wanted),query_policy='All non-generic exact public references at original TRAIN t=0 only; original yaw; no destination GT/future observation')
 if a.output.exists():
  if not a.resume or json.loads((a.output/'fingerprint.json').read_text())!=fingerprint:raise ValueError('Refuse changed/incomplete cache overwrite')
 else:a.output.mkdir(parents=True);atomic_json(fingerprint,a.output/'fingerprint.json')
 assert not (a.output/'manifest.json').exists();(a.output/'observations').mkdir(exist_ok=True)
 oldproof=json.loads((a.old_cache/'manifest.json').read_text());assert file_sha256(a.old_cache/'cache.pt')==oldproof['cache_sha256'];t=time.perf_counter();old=torch.load(a.old_cache/'cache.pt',weights_only=True,map_location='cpu');old_lookup={}
 for f in old['frames']:
  depth=Path(f['image_path']).parent/Path(f['frame']['depth']).name
  if not depth.exists():continue
  with np.load(depth) as z:key=(f['frame']['image_sha256'],digest_bytes(z['depth_valid'].tobytes()))
  old_lookup[key]=f['image_features']
 old_lookup_seconds=time.perf_counter()-t;vectors={}
 for q in old['queries'].values():vectors[q['name']]=q['name_text'];vectors[q['type']]=q['type_text']
 del old;gc.collect();frames=[];records=[];queries={};begin=0;chunks=[]
 if a.resume:
  progress=json.loads((a.output/'progress.json').read_text());begin=progress['next_inventory'];chunks=progress['chunks'];excluded.update(progress['excluded']);metrics=progress['timings']
  for ch in chunks:
   value=torch.load(a.output/ch['file'],map_location='cpu',weights_only=True);assert file_sha256(a.output/ch['file'])==ch['sha256'];frames+=value['frames'];records+=value['records'];queries.update(value['queries'])
 t=time.perf_counter();native=ObservedSigLIPMatchingEncoder('google/siglip2-large-patch16-384',device='cuda',microbatch=1);torch.cuda.synchronize();cold_encoder_seconds=time.perf_counter()-t;renderer=CityNavObservedViewRenderer(a.data_root/'rgbd',a.output/'observations',384);new_features={};cache_hits=0;vision_frames=0;f0=len(frames);r0=len(records);query_saved=set(queries);last=len(wanted)-1
 for pos in range(begin,len(wanted)):
  task=wanted[pos];mp=task['map_name'];t=time.perf_counter();obs=renderer.current_rgbd(mp,task['pose'],GROUND_LEVEL[mp]);render_s=time.perf_counter()-t;eligible=[];s=time.perf_counter()
  for name in sorted(task['names']):
   q=named[mp][name];anchor=local_landmark_anchor(q['contours'],obs)
   if anchor['train_eligible'] and int(anchor['valid_patches'].sum())>=8:eligible.append((q,anchor))
  geometry_s=time.perf_counter()-s
  if not eligible:excluded['no_valid_reliable_local_anchor_frame']+=1
  else:
   _,png=cv2.imencode('.png',obs['bgr']);image_sha=digest_bytes(png.tobytes());depth_sha=digest_bytes(obs['depth_valid'].tobytes());featurekey=(image_sha,depth_sha);bounds=[renderer._raster.bounds.left,renderer._raster.bounds.bottom,renderer._raster.bounds.right,renderer._raster.bounds.top];t=time.perf_counter();origin='new';features=old_lookup.get(featurekey,new_features.get(featurekey))
   if features is not None:
    cache_hits+=1;origin='reused_exact_RGB_and_depth'
   else:
    q=eligible[0][0];inputs,reason=encode_landmark_observation(obs,dict(name=q['name'],typ=q['type'],contours=q['contours']),dict(world_bounds=bounds),native,torch.device('cuda'));assert inputs is not None,reason;torch.cuda.synchronize();features=inputs['image_features'][0].detach().cpu().half().clone();new_features[featurekey]=features;vision_frames+=1
   encode_s=time.perf_counter()-t
   # Geometry/validity independent of the requested name; no repeated RGB tower.
   from graphnav.native_dense_regions import native_dense_regions,sparse_view
   sparse=sparse_view(native_dense_regions(obs));world=np.asarray(sparse['world'],np.float32);uv=(world-np.asarray(bounds[:2]))/(np.asarray(bounds[2:])-bounds[:2]);valid=np.asarray(sparse['valid'],bool)&(np.asarray(sparse['quality'])[:,0]>=.9)&np.isfinite(uv).all(-1)&(uv>=0).all(-1)&(uv<=1).all(-1)
   if not valid.any():excluded['no_valid_world_patches']+=1;continue
   fid=len(frames);key=f'train_t0_{mp}_{pos:05d}';img=a.output/'observations'/f'{key}.png';depth=a.output/'observations'/f'{key}.npz';img.write_bytes(png.tobytes());np.savez_compressed(depth,depth_valid=obs['depth_valid']);frame=dict(frame_key=key,map_name=mp,image_sha256=image_sha,image=img.name,depth=depth.name,footprint_world=np.asarray(obs['footprint_world']).tolist(),camera_pose=task['pose'],source_split='train_seen',source_episodes=task['source_episodes'],step=0)
   global_world=torch.tensor(bounds[:2])+grid_centers(28)*torch.tensor(np.asarray(bounds[2:])-bounds[:2]);frames.append(dict(image_features=features,local_uv=torch.from_numpy(uv).float(),valid=torch.from_numpy(valid),global_world=global_world.float(),bounds=bounds,image_path=str(img),frame=frame,map_name=mp,city=mp.split('_block_')[0],subset='fit' if mp.startswith('cambridge_') else 'dev'))
   for q,anchor in eligible:
    if q['key'] not in queries:
     for text in [q['name'],q['type']]:
      if text not in vectors:vectors[text]=native.texts([text]).detach().cpu().half().clone()
     queries[q['key']]={k:v for k,v in q.items() if k!='region'};queries[q['key']].update(name_text=vectors[q['name']],type_text=vectors[q['type']])
    # Coverage labels are geometric proxies, not CityNav target/visibility GT.
    dx,dy=(np.asarray(bounds[2:])-bounds[:2])/28;g=global_world.numpy();cells=shapely.box(g[:,0]-dx/2,g[:,1]-dy/2,g[:,0]+dx/2,g[:,1]+dy/2);area=shapely.area(shapely.intersection(cells,q['region']))/(dx*dy)
    records.append(dict(frame_id=fid,query_key=q['key'],type=q['type'],subset=frames[-1]['subset'],labels=torch.tensor(anchor['labels']).long(),polygon_tokens=torch.from_numpy(local_anchor_tokens(q['contours'],obs['footprint_world'])),distance=torch.from_numpy(shapely.distance(shapely.points(g),q['region']).astype(np.float32)),area_fraction=torch.from_numpy(area.astype(np.float32)),visibility='geometry_proxy_unknown',hard_negative_indices=[]))
   metrics.append(dict(inventory=pos,render_seconds=render_s,local_supervision_seconds=geometry_s,encode_and_pool_seconds=encode_s,feature_origin=origin,queries=len(eligible)))
  if (pos+1)%128==0 or pos==last:
   path=a.output/f'chunk_{len(chunks):05d}.pt';qnew={k:v for k,v in queries.items() if k not in query_saved};atomic_save(dict(frames=frames[f0:],records=records[r0:],queries=qnew),path);chunks.append(dict(file=path.name,sha256=file_sha256(path)));f0=len(frames);r0=len(records);query_saved=set(queries);atomic_json(dict(next_inventory=pos+1,chunks=chunks,excluded=dict(excluded),timings=metrics),a.output/'progress.json');print(json.dumps(dict(inventory=pos+1,total=len(wanted),frames=len(frames),queries=len(records),vision_frames=vision_frames,reuse=cache_hits,seconds=time.perf_counter()-start)),flush=True)
 atomic_save(dict(frames=frames,records=records,queries=queries),a.output/'cache.pt');counts={s:dict(frames=sum(f['subset']==s for f in frames),queries=sum(r['subset']==s for r in records),types=dict(Counter(r['type'] for r in records if r['subset']==s))) for s in ['fit','dev']};atomic_json(dict(complete=True,cache_version='unified-v2-full-TRAIN-original-t0',counts=counts,cache_sha256=file_sha256(a.output/'cache.pt'),source= fingerprint,seconds=time.perf_counter()-start,cold_encoder_seconds=cold_encoder_seconds,old_lookup_seconds=old_lookup_seconds,unique_frames=len(frames),query_count=len(records),vision_frames_this_session=vision_frames,exact_cache_hits_this_session=cache_hits,excluded=dict(excluded),timings=metrics,supervision='Public Polygon weak region at original current t0; no target GT, no future/teacher trajectory as network input',cache_note='One visual forward per unique RGB/depth mask; queries share it; original yaw unchanged; frame/key/split explicit'),a.output/'manifest.json');print(json.dumps(counts),flush=True)
if __name__=='__main__':main()
