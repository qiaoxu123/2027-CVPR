"""Post-hoc TRAIN footprint/depth support along actual visited decision poses.

These are geometric proxies, not verified target visibility/instance matches.
Never invokes a model or changes an action, Goal, Stop or saved trajectory.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.renderer import CityNavObservedViewRenderer
from graphnav.fixed_yaw_observation import FixedYawObservation
from graphnav.landmark_projection import world_to_frame
from graphnav.local_chain_observation import local_patch_regions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('run','data-root','sbf-source','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.sbf_source))
    from sbfnav.dataset import CityNavDataset
    from sbfnav.mapdata import GROUND_LEVEL
    dataset=CityNavDataset(a.data_root,'train_seen',version='processed_citynav')
    if file_sha256(dataset.annotation_path)!=json.loads((a.run/'run_manifest.json').read_text())['train_annotation_sha256']:
        raise ValueError('TRAIN labels changed after rollout')
    renderer=FixedYawObservation(CityNavObservedViewRenderer(a.data_root/'rgbd',a.output.parent/'unused',image_size=384),0.)
    summaries={}
    for method in ('original','real'):
        details=[]
        for row in map(json.loads,(a.run/method/'predictions.jsonl').read_text().splitlines()):
            record=dataset[row['index']];gt=record.target_position[:2];states=[]
            for step in range(len(row['step_diagnostics'])):
                obs=renderer.current_rgbd(record.map_name,row['poses'][step],GROUND_LEVEL[record.map_name])
                pixel=world_to_frame(np.array([gt]),obs['footprint_world'],obs['bgr'].shape)[0]
                inside=bool(((pixel>=0)&(pixel<=383)).all());depth=False
                if inside:
                    x,y=np.rint(pixel).astype(int);depth=bool(obs['depth_valid'][y,x])
                world,_,valid,_=local_patch_regions(obs)
                spatial=bool(valid.any() and np.linalg.norm(world[valid]-gt,axis=-1).min()<=20.)
                states.append(dict(step=step,target_point_in_footprint=inside,target_point_depth_supported=depth,
                                   valid_local_position_within20m=spatial))
            details.append(dict(index=record.index,success=bool(row['metrics']['SR']),
                never_correct_goal=not row['posthoc_goal_analysis']['ever_correct'],
                ever_footprint=any(s['target_point_in_footprint'] for s in states),
                ever_depth_supported=any(s['target_point_depth_supported'] for s in states),
                ever_spatial_support20m=any(s['valid_local_position_within20m'] for s in states),states=states))
        failed=[r for r in details if not r['success']];never=[r for r in failed if r['never_correct_goal']]
        summaries[method]=dict(episodes=len(details),failures=len(failed),never_correct_failures=len(never),
            never_correct_with_geometric_view=sum(r['ever_footprint'] for r in never),
            never_correct_with_depth_proxy=sum(r['ever_depth_supported'] for r in never),
            never_correct_with_local_spatial_support=sum(r['ever_spatial_support20m'] for r in never),details=details)
    atomic_json(dict(scope='Geometric/depth proxies only, not instance matches or verified visibility',methods=summaries),a.output)
    print(json.dumps({m:{k:v for k,v in s.items() if k!='details'} for m,s in summaries.items()},indent=2))


if __name__=='__main__':main()
