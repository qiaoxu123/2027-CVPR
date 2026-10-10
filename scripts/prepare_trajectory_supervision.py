"""Build label-separated real CityFlight waypoint/Stop samples, including recovery poses."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
import numpy as np
from graphnav.closedloop import PublicEpisode
from graphnav.a3_runtime import public_static_record
from graphnav.checkpointing import file_sha256
from graphnav.trajectory_policy import remaining_waypoints


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sbf-source',required=True);p.add_argument('--data-root',required=True)
    p.add_argument('--allowed-train-queries',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--max-episodes',type=int)
    args=p.parse_args();sys.path.insert(0,args.sbf_source)
    from sbfnav.dataset import CityNavDataset,CityReferCatalog
    from sbfnav.coordinates import MapTransform
    from sbfnav.mapdata import GROUND_LEVEL
    out=Path(args.output)
    if out.exists():raise ValueError('refusing overwrite')
    out.mkdir(parents=True);root=Path(args.data_root)
    data=CityNavDataset(root,'train_seen');catalog=CityReferCatalog(root/'cityrefer')
    allowed={(r['map_name'],r['instruction']) for r in map(json.loads,Path(args.allowed_train_queries).read_text().splitlines())}
    transforms={};queries={};counts=Counter();rng=random.Random(args.seed)
    with (out/'samples.jsonl').open('w') as samples:
        for index,record in enumerate(data):
            if args.max_episodes is not None and index>=args.max_episodes:break
            if (record.map_name,record.instruction) not in allowed:
                counts['excluded_nontraining_query']+=1;continue
            if record.map_name not in transforms:
                transforms[record.map_name]=MapTransform.from_raster(root/'rgbd'/f'{record.map_name}.tif')
            transform=transforms[record.map_name];ground=GROUND_LEVEL[record.map_name]
            public=PublicEpisode(record.map_name,record.instruction,record.start_pose,(),
                tuple(catalog.referenced_landmark_names(record)))
            query=public_static_record(public,transform,catalog)
            signature=json.dumps(query,sort_keys=True)
            if signature not in queries:queries[signature]=len(queries)
            query_index=queries[signature]
            prefix_indices=sorted(set([0,len(record.trajectory)//3,2*len(record.trajectory)//3,len(record.trajectory)-1]))
            distances=np.linalg.norm(record.trajectory[:,:2]-record.target_position[None,:2],axis=1)
            near_negative=np.flatnonzero((distances>20)&(distances<=40))
            if len(near_negative):
                prefix_indices=sorted(set(prefix_indices+[int(near_negative[np.argmin(distances[near_negative])])]))
            recovery_step=prefix_indices[1] if len(prefix_indices)>1 else None
            for step in prefix_indices:
                source_pose=record.trajectory[step].copy()
                teacher=remaining_waypoints(record.trajectory,step)
                for recovery in (False,True) if step==recovery_step else (False,):
                    pose=source_pose.copy()
                    if recovery:
                        angle=rng.random()*2*np.pi;distance=rng.uniform(10.,30.)
                        pose[:2]+=distance*np.array([np.cos(angle),np.sin(angle)])
                        pose[0]=np.clip(pose[0],transform.x_min,transform.x_max)
                        pose[1]=np.clip(pose[1],transform.y_min,transform.y_max)
                        pose[3]=rng.uniform(-np.pi,np.pi)
                        # The first recovery supervision is the actual source pose;
                        # following points are its genuine remaining human route.
                        future=np.concatenate([source_pose[None,:3],teacher[:7]],axis=0)
                    else:future=teacher
                    distance=float(np.linalg.norm(pose[:2]-record.target_position[:2]))
                    row=dict(query_index=query_index,source_episode_index=record.index,
                        source_step=step,recovery=recovery,pose_world=pose.tolist(),
                        initial_pose_world=record.start_pose.tolist(),ground_level=ground,
                        # No images are supplied here: recovery observations must
                        # later be rendered from the perturbed pose, never copied from teacher.
                        observation_source='not_rendered; geometry/pose baseline only',
                        labels=dict(target_world=record.target_position.tolist(),teacher_world=future.tolist(),
                            valid_teacher_z=((future[:,2]>=ground+10)&(future[:,2]<=ground+100)).tolist(),
                            stop=distance<=20.,goal_distance_m=distance))
                    samples.write(json.dumps(row)+'\n');counts['samples']+=1
                    counts['recovery_samples']+=int(recovery);counts['positive_stop_samples']+=int(distance<=20.)
                    counts['near_false_stop_samples_20_40m']+=int(20.<distance<=40.)
                    counts['out_of_controller_altitude_samples']+=int(not ground+10<=pose[2]<=ground+100)
            counts['episodes']+=1
    with (out/'public_queries.jsonl').open('w') as f:
        for signature in queries:f.write(signature+'\n')
    manifest=dict(counts=counts,queries=len(queries),args=vars(args),seed=args.seed,
        input_sha256=dict(source_train=file_sha256(data.annotation_path),objects=file_sha256(root/'cityrefer/objects.json'),
            processed=file_sha256(root/'cityrefer/processed_descriptions.json'),allowed_query_file=file_sha256(args.allowed_train_queries)),
        output_sha256={name:file_sha256(out/name) for name in ['samples.jsonl','public_queries.jsonl']},
        protocol='Source annotation labels stored separately; true target/expert future never used as inference inputs. Recovery connector is explicit augmentation; future rows are real human poses. No synthetic RGB. Invalid-altitude teacher z has a separate loss mask rather than a fabricated clipped label.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))


if __name__=='__main__':main()
