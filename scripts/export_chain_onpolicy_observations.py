"""Reconstruct only actual TRAIN-fit decision observations, verify runtime SHA."""
import argparse, hashlib, json, sys, time
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from graphnav.checkpointing import file_sha256
from graphnav.renderer import CityNavObservedViewRenderer
from graphnav.fixed_yaw_observation import FixedYawObservation
from graphnav.local_chain_observation import local_patch_regions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('run','fit-labels','data-root','sbf-source','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();started=time.perf_counter()
    if a.output.exists():raise FileExistsError('preserve previous acquisition')
    fit={json.loads(x)['task_key'] for x in a.fit_labels.read_text().splitlines()}
    report=json.loads((a.run/'real/metrics.json').read_text())
    if report['split']!='train_seen':raise ValueError('TRAIN only')
    rows=[json.loads(x) for x in (a.run/'real/predictions.jsonl').read_text().splitlines()]
    if len(rows)!=report['episodes']:raise ValueError('incomplete rollout')
    if len({x['index'] for x in rows})!=len(rows):raise ValueError('duplicate episodes')
    if any(f"train_{x['index']:05d}" not in fit for x in rows):raise ValueError('non-fit target prohibited')
    sys.path.insert(0,str(a.sbf_source))
    from sbfnav.dataset import CityNavDataset,CityReferCatalog
    from sbfnav.mapdata import GROUND_LEVEL
    ds=CityNavDataset(a.data_root,'train_seen',version='processed_citynav');cat=CityReferCatalog(a.data_root/'cityrefer')
    a.output.mkdir(parents=True);(a.output/'rgb').mkdir()
    renderer=FixedYawObservation(CityNavObservedViewRenderer(a.data_root/'rgbd',a.output/'unused',image_size=384),0.)
    public=[];labels=[];tasks=[];stats=Counter()
    for row in rows:
        rec=ds[row['index']];key=f"train_{row['index']:05d}";obj=cat.objects[rec.map_name][rec.object_id]
        tasks.append(dict(task_key=key,episode_index=row['index'],map_name=rec.map_name,category=obj['object_type'],physical_group=[rec.map_name,rec.object_id],instruction=rec.instruction))
        for step,diag in enumerate(row['step_diagnostics']):
            pose=row['poses'][step];obs=renderer.current_rgbd(rec.map_name,pose,GROUND_LEVEL[rec.map_name]);im=obs['bgr']
            h=hashlib.sha256(str(im.shape).encode());h.update(np.ascontiguousarray(im).tobytes())
            if h.hexdigest()!=diag['observed_rgb_sha256']:raise ValueError('input reconstruction mismatch')
            frame=f'{key}_onpolicy_{step:04d}';image=Path('rgb')/(frame+'.png');depth=Path('rgb')/(frame+'_depth.npz')
            if not cv2.imwrite(str(a.output/image),im):raise IOError(image)
            np.savez_compressed(a.output/depth,depth_m=obs['depth_m'],depth_valid=obs['depth_valid'])
            public.append(dict(frame_key=frame,task_key=key,map_name=rec.map_name,instruction=rec.instruction,step=step,agent_pose=pose,camera_pose=np.asarray(obs['pose']).tolist(),footprint_world=np.asarray(obs['footprint_world']).tolist(),image=str(image),image_sha256=file_sha256(a.output/image),depth=str(depth),depth_sha256=file_sha256(a.output/depth),runtime_pixel_sha256=h.hexdigest(),valid_fraction=float(obs['depth_valid'].mean()),source='actual frozen TRAIN-fit policy decision; no teacher poses'))
            world,_,valid,_=local_patch_regions(obs)
            near=bool(valid.any() and (np.linalg.norm(world[valid]-rec.target_position[:2],axis=-1)<=20).any())
            stats['near_goal20_world_support' if near else 'no_goal20_world_support']+=1
            labels.append(dict(frame_key=frame,task_key=key,status='captured',target_world=rec.target_position.tolist(),category=obj['object_type'],physical_group=[rec.map_name,rec.object_id]))
    manifest=dict(complete=True,split='train_seen',source='actual TRAIN-fit closed-loop decisions; input pixels verified',fixed_camera_yaw_rad=0.,frames=len(public),tasks=len(tasks),maps=len({x['map_name'] for x in tasks}),fit_labels_sha256=file_sha256(a.fit_labels),run_manifest_sha256=file_sha256(a.run/'run_manifest.json'),world_support_proxy=dict(stats),warning='20m world support is not visible-instance supervision; private labels excluded from feature extraction')
    for name,values in [('public_frames.jsonl',public),('labels_private.jsonl',labels),('tasks_private.json',tasks)]:
        path=a.output/name;path.write_text(json.dumps(values,indent=2) if name.endswith('.json') else ''.join(json.dumps(x)+'\n' for x in values));manifest[name+'_sha256']=file_sha256(path)
    manifest['seconds']=time.perf_counter()-started;(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))


if __name__=='__main__':main()
