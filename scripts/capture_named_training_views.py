"""Public landmark training views sampled offline; never a navigation result."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch
from graphnav.checkpointing import atomic_json,atomic_save,file_sha256
from graphnav.landmark_alias import canonical_name
from graphnav.landmark_region_supervision import named_union,public_type
from graphnav.renderer import CityNavObservedViewRenderer
from graphnav.fixed_yaw_observation import FixedYawObservation
from graphnav.native_dense_regions import native_dense_regions,sparse_view
from graphnav.appearance_matching import ObservedSigLIPMatchingEncoder
from graphnav.geometry import grid_centers
from scripts.run_native_grid_pilot import pool
from scripts.analyze_instruction_landmark_association import PublicNames,iter_train


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('dataset-profile','map-manifest','catalog','data-root','sbf-source','backbone','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    sys.path.insert(0,str(a.sbf_source))
    from sbfnav.coordinates import MapTransform
    from sbfnav.mapdata import GROUND_LEVEL
    profile=json.loads(a.dataset_profile.read_text());assignment=json.loads(a.map_manifest.read_text())['map_assignment']
    catalog=json.loads(a.catalog.read_text());names=PublicNames(catalog);candidates={}
    for i,row in iter_train(Path(profile['processed_all_annotations'])):
        mp=f"{row['area']}_block_{row['block']}";text=row['descriptions'][0]
        for mention in names.match(mp,text):
            for name in mention['public_names']:
                key=mp+'::'+canonical_name(name)
                if key in candidates:continue
                group=[(oid,obj) for oid,obj in catalog[mp].items() if canonical_name(obj.get('name',''))==canonical_name(name)]
                typ=public_type([obj['object_type'] for _,obj in group])
                if typ not in {'Building','Road'}:continue
                if len(group)>16:continue
                try:region=named_union([obj['contour'] for _,obj in group])
                except ValueError:continue
                if key not in candidates:
                    candidates[key]=dict(query_key=key,map_name=mp,name=name,type=typ,source_instruction=text,subset=assignment[mp],
                        public_ids=[oid for oid,_ in group],point=list(region.representative_point().coords[0]))
    selected=[]
    for subset,per_type in [('fit',75),('dev',16),('confirm',16)]:
        for typ in ('Building','Road'):
            maps=defaultdict(list)
            for r in candidates.values():
                if r['subset']==subset and r['type']==typ:maps[r['map_name']].append(r)
            for mp in maps:maps[mp].sort(key=lambda r:hashlib.sha256(('0:'+r['query_key']).encode()).hexdigest())
            chosen=[]
            while len(chosen)<per_type and any(maps.values()):
                for mp in sorted(maps):
                    if maps[mp] and len(chosen)<per_type:chosen.append(maps[mp].pop(0))
            selected+=chosen
    selected.sort(key=lambda r:(r['map_name'],r['query_key']))
    a.output.mkdir(parents=True);cache=a.output/'features';obs=a.output/'observations';cache.mkdir();obs.mkdir()
    atomic_json(selected,a.output/'selection_public.json')
    renderer=FixedYawObservation(CityNavObservedViewRenderer(a.data_root/'rgbd',obs,image_size=384),0.)
    native=ObservedSigLIPMatchingEncoder(str(a.backbone));torch.set_num_threads(4)
    frames=[];records=[];chunks=[];all_keys=[];started=time.perf_counter();uv=grid_centers(28).numpy()
    def flush():
        if not records:return
        file=f'chunk_{len(chunks):05d}.pt';atomic_save(dict(records=list(records)),cache/file)
        chunks.append(dict(file=file,sha256=file_sha256(cache/file),rows=len(records)));records.clear()
    with torch.inference_mode():
        for qi,q in enumerate(selected):
            mp=q['map_name'];transform=MapTransform.from_raster(a.data_root/'rgbd'/f'{mp}.tif')
            bounds=np.array([transform.x_min,transform.y_min,transform.x_max,transform.y_max],np.float32)
            global_xy=bounds[:2]+uv*(bounds[2:]-bounds[:2])
            task='named_train_'+hashlib.sha256(q['query_key'].encode()).hexdigest()[:16]
            for step,(dx,dy,height) in enumerate([(-25,0,50),(25,0,50),(0,-25,80),(0,25,80)]):
                xy=np.clip(np.array(q['point'])+[dx,dy],bounds[:2]+.1,bounds[2:]-.1)
                pose=np.array([*xy,GROUND_LEVEL[mp]+height,0.],float)
                view=renderer.current_rgbd(mp,pose,GROUND_LEVEL[mp]);key=f'{task}_{step:02d}'
                image=f'{key}.png';depth=f'{key}.npz'
                cv2.imwrite(str(obs/image),view['bgr']);np.savez_compressed(obs/depth,depth_valid=view['depth_valid'])
                dense=native_dense_regions(dict(bgr=view['bgr'],depth_valid=view['depth_valid'],footprint_world=view['footprint_world']))
                region=sparse_view(dense)
                pixels=native.processor(images=[cv2.cvtColor(view['bgr'],cv2.COLOR_BGR2RGB)],return_tensors='pt').to('cuda')
                with torch.autocast('cuda',dtype=torch.bfloat16):patches=native.model.vision_model(**pixels).last_hidden_state[0]
                features=pool(native,patches,region['masks'])
                frame=dict(frame_key=key,task_key=task,map_name=mp,instruction=q['source_instruction'],
                    requested_public_name=q['name'],step=step,agent_pose=pose.tolist(),camera_pose=pose.tolist(),
                    footprint_world=view['footprint_world'].tolist(),image=image,image_sha256=file_sha256(obs/image),
                    depth=depth,depth_sha256=file_sha256(obs/depth),valid_fraction=float(view['depth_valid'].mean()),
                    view_source='offline_public_landmark_pose_sampling_not_human_or_predicted_navigation')
                frames.append(frame);all_keys.append(key)
                records.append(dict(frame_key=key,task_key=task,map_name=mp,subset=q['subset'],step=step,
                    local_uv=torch.from_numpy((region['world']-bounds[:2])/(bounds[2:]-bounds[:2])),
                    candidate_world=torch.from_numpy(np.concatenate([global_xy,region['world']])),
                    image_features=features,local_valid=torch.from_numpy(region['valid']),quality=torch.from_numpy(region['quality'])))
                if len(records)>=8:flush()
            if (qi+1)%20==0:print(json.dumps(dict(names=qi+1,total=len(selected),frames=len(frames),seconds=time.perf_counter()-started)),flush=True)
    flush()
    (obs/'public_frames.jsonl').write_text(''.join(json.dumps(f)+'\n' for f in frames))
    atomic_json(dict(frame_keys=all_keys,public_pose_sampling=True),a.output/'public_filter.json')
    manifest=dict(complete=True,split='train_seen',camera_yaw_rad=0.,chunks=chunks,frames=len(frames),
        named_groups=len(selected),selection_counts=dict(Counter(q['subset']+'::'+q['type'] for q in selected)),
        seconds=time.perf_counter()-started,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
        dataset_profile_sha256=file_sha256(a.dataset_profile),source_map_manifest_sha256=file_sha256(a.map_manifest),
        public_catalog_sha256=file_sha256(a.catalog),selection_sha256=file_sha256(a.output/'selection_public.json'),
        frames_sha256=file_sha256(obs/'public_frames.jsonl'),producer_sha256=file_sha256(__file__),
        warning='Public TRAIN landmark pose sampling and real orthophoto crops only; not human trajectories, future inference observations, pixel GT, or closed-loop SR.')
    atomic_json(manifest,cache/'manifest.json');atomic_json(manifest,a.output/'manifest.json');print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':main()
