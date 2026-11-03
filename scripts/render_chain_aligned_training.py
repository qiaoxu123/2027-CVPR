"""TRAIN-fit only counterfactual action-height views, not visited observations."""
import argparse,json,sys,time
from pathlib import Path
import cv2
import numpy as np
from graphnav.deployment_observation_filter import observation_in_envelope,aligned_observation_pose
from graphnav.renderer import CityNavObservedViewRenderer
from graphnav.fixed_yaw_observation import FixedYawObservation
from graphnav.checkpointing import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('observations','fit-labels','data-root','sbf-source','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve previous observations')
    sys.path.insert(0,str(a.sbf_source))
    from sbfnav.mapdata import GROUND_LEVEL
    frames=[json.loads(x) for x in (a.observations/'public_frames.jsonl').read_text().splitlines()]
    labels={x['frame_key']:x for x in map(json.loads,a.fit_labels.read_text().splitlines())}
    chosen=[f for f in frames if f['frame_key'] in labels and not observation_in_envelope(f,GROUND_LEVEL[f['map_name']])]
    chosen.sort(key=lambda f:(f['map_name'],f['task_key'],f['step']))
    a.output.mkdir(parents=True);(a.output/'rgb').mkdir();started=time.perf_counter()
    renderer=FixedYawObservation(CityNavObservedViewRenderer(a.data_root/'rgbd',a.output/'unused',image_size=384),0.)
    public=[];private=[]
    for f in chosen:
        source=f['agent_pose'];pose=aligned_observation_pose(f,GROUND_LEVEL[f['map_name']])
        if pose[:2]!=source[:2] or pose[3]!=source[3] or pose==source:raise ValueError('only noninitial height can change')
        obs=renderer.current_rgbd(f['map_name'],pose,GROUND_LEVEL[f['map_name']]);key=f['frame_key']+'_aligned'
        image=Path('rgb')/(key+'.png');depth=Path('rgb')/(key+'_depth.npz')
        if not cv2.imwrite(str(a.output/image),obs['bgr']):raise IOError(image)
        np.savez_compressed(a.output/depth,depth_m=obs['depth_m'],depth_valid=obs['depth_valid'])
        row=dict(frame_key=key,task_key=f['task_key'],map_name=f['map_name'],instruction=f['instruction'],step=f['step'],agent_pose=pose,camera_pose=np.asarray(obs['pose']).tolist(),footprint_world=np.asarray(obs['footprint_world']).tolist(),image=str(image),image_sha256=file_sha256(a.output/image),depth=str(depth),depth_sha256=file_sha256(a.output/depth),valid_fraction=float(obs['depth_valid'].mean()),source_frame_key=f['frame_key'],source_agent_pose=source,source='Counterfactual TRAIN reference XY at action-compatible height; NOT an actual closed-loop visited frame')
        public.append(row);private.append(dict(labels[f['frame_key']],frame_key=key))
    manifest=dict(complete=True,split='train_seen',source='Simulated TRAIN-fit action-height views; original XY unchanged, not actual visited observations',fixed_camera_yaw_rad=0.,frames=len(public),maps=len({x['map_name'] for x in public}),source_public_frames_sha256=file_sha256(a.observations/'public_frames.jsonl'),fit_labels_sha256=file_sha256(a.fit_labels),seconds=time.perf_counter()-started)
    for name,rows in [('public_frames.jsonl',public),('labels_private.jsonl',private)]:
        path=a.output/name;path.write_text(''.join(json.dumps(x)+'\n' for x in rows));manifest[name+'_sha256']=file_sha256(path)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))


if __name__=='__main__':main()
