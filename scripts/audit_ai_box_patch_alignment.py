"""TRAIN-only AI weak box versus actual native patch lattice and goal labels."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.landmark_projection import frame_to_world
from graphnav.local_chain_observation import local_patch_regions
from scripts.ai_weak_box_schema import validate_weak_box_record


def center_hits(points,box):
    x0,y0,x1,y1=box
    return (points[:,0]>=x0)&(points[:,0]<x1)&(points[:,1]>=y0)&(points[:,1]<y1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','boxes','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    boxes=json.loads(a.boxes.read_text());keys={x['frame_key'] for x in boxes};frames={};truth={}
    for obs in ('fixed_yaw_train400_s1','scoped_between_train294_observations_v1'):
        for f in map(json.loads,(a.root/obs/'public_frames.jsonl').read_text().splitlines()):
            if f['frame_key'] in keys:frames[f['frame_key']]=(f,a.root/obs)
        for f in map(json.loads,(a.root/obs/'labels_private.jsonl').read_text().splitlines()):
            if f.get('frame_key') in keys:truth[f['frame_key']]=f
    g=np.arange(24);gy,gx=np.meshgrid(g,g,indexing='ij');dense=(np.stack([gx.ravel(),gy.ravel()],axis=-1)+.5)*16-.5
    sparse_ids=np.array([y*24+x for y in range(1,24,2) for x in range(1,24,2)]);sparse=dense[sparse_ids]
    records=[]
    for record in boxes:
        key=record['frame_key'];f,obs=frames[key];image=obs/f['image'];assert file_sha256(image)==f['image_sha256']
        validate_weak_box_record(record,f['image_sha256'])
        if not record['boxes_px']:continue
        box=np.asarray(record['boxes_px'][0],float);bgr=cv2.imread(str(image))
        depth=obs/f['depth'];assert file_sha256(depth)==f['depth_sha256']
        with np.load(depth) as d:valid=d['depth_valid'].copy()
        support=(bgr.max(-1)>0)&valid;dense_valid=(support.reshape(24,16,24,16).mean((1,3))>=.9).ravel()
        _,_,native_valid,_=local_patch_regions(dict(bgr=bgr,depth_valid=valid,footprint_world=f['footprint_world']))
        assert np.array_equal(native_valid,dense_valid[sparse_ids])
        hit144=center_hits(sparse,box);hit576=center_hits(dense,box)
        # Center hits are grid coverage proxies, NOT detector or instance recall.
        center=box.reshape(2,2).mean(0);world=frame_to_world(center[None],f['footprint_world'],bgr.shape)[0]
        target=np.asarray(truth[key]['target_world'][:2]);goal_error=float(np.linalg.norm(world-target))
        sensitivity={}
        for delta in (-2,2):
            adjusted=box+np.asarray([-delta,-delta,delta,delta]);adjusted[[0,2]]=adjusted[[0,2]].clip(0,384);adjusted[[1,3]]=adjusted[[1,3]].clip(0,384)
            sensitivity[str(delta)]=dict(valid144=bool((center_hits(sparse,adjusted)&native_valid).any()),valid576=bool((center_hits(dense,adjusted)&dense_valid).any()))
        records.append(dict(boundary_sensitivity=sensitivity,frame_key=key,width_px=float(box[2]-box[0]),height_px=float(box[3]-box[1]),
            raw144_center=bool(hit144.any()),valid144_center=bool((hit144&native_valid).any()),
            raw576_center=bool(hit576.any()),valid576_center=bool((hit576&dense_valid).any()),
            ai_box_center_world=world.tolist(),registered_goal_error_m=goal_error,
            source_image_sha256=f['image_sha256'],human_verified=False,pixel_gold=False))
    a.output.mkdir(parents=True);summary=dict(boxes=len(records),
        **{k:sum(x[k] for x in records) for k in ('raw144_center','valid144_center','raw576_center','valid576_center')},
        median_width_px=float(np.median([x['width_px'] for x in records])),median_height_px=float(np.median([x['height_px'] for x in records])),
        goal_error_median_m=float(np.median([x['registered_goal_error_m'] for x in records])),
        goal_error_max_m=max(x['registered_goal_error_m'] for x in records),goal_error_over20=sum(x['registered_goal_error_m']>20 for x in records),
        boundary_sensitivity={str(delta):{method:sum(x['boundary_sensitivity'][str(delta)][method] for x in records) for method in ('valid144','valid576')} for delta in (-2,2)},
        inference_modified=False,used_for_training=False,annotation_sha256=file_sha256(a.boxes),producer_sha256=file_sha256(__file__),
        warning='AI weak appearance boxes authored with TRAIN annotation hints. Grid coverage and registration diagnostics only; not pixel gold, blind instance accuracy, Goal R@1 or SR.')
    atomic_json(records,a.output/'records.json');atomic_json(summary,a.output/'summary.json');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
