"""TRAIN geometry-only radius/grid support; never an instance recall claim."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.proposal_grid_audit import observed_grid_sites
from graphnav.training_retarget_proposals import registered_core_proposals


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('observations','feature-cache','filter','data-root','sbf-source','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError('Preserve earlier support audit')
    manifest=json.loads((args.feature_cache/'manifest.json').read_text())
    if manifest['split']!='train_seen': raise ValueError('TRAIN only')
    sys.path.insert(0,str(args.sbf_source))
    from sbfnav.dataset import CityNavDataset
    dataset=CityNavDataset(args.data_root,'train_seen',version='processed_citynav')
    catalog=json.loads((args.data_root/'cityrefer/objects.json').read_text())
    registry={m:defaultdict(list) for m,s in manifest['map_assignment'].items() if s=='fit'}
    for index,raw in enumerate(dataset._raw_records):
        if raw['split']!='train_seen': raise ValueError('non-TRAIN record')
        name=f"{raw['area']}_block_{raw['block']}";oid=str(raw['object_ids'][0])
        if name in registry and catalog[name][oid]['object_type']=='Car':registry[name][oid].append(index)
    objects={name:(sorted(table),np.array([catalog[name][oid]['position'][:2] for oid in sorted(table)],float).reshape(-1,2)) for name,table in registry.items()}
    allowed=set(json.loads(args.filter.read_text())['frame_keys'])
    frames=[r for r in map(json.loads,(args.observations/'public_frames.jsonl').read_text().splitlines()) if r['frame_key'] in allowed and manifest['map_assignment'][r['map_name']]=='fit']
    counters=defaultdict(Counter);groups=defaultdict(set); strict=[]
    # Priors fixed before this audit: 1m cores and <=4m native patch width
    # to reduce adjacent-car mixing. This is TRAIN annotation selection only.
    for frame in frames:
        image_path=args.observations/frame['image']; depth_path=args.observations/frame['depth']
        if file_sha256(image_path)!=frame['image_sha256'] or file_sha256(depth_path)!=frame['depth_sha256']:raise ValueError('RGB/depth changed')
        image=cv2.imread(str(image_path)); depth=np.load(depth_path)
        obs=dict(bgr=image,depth_valid=depth['depth_valid'],footprint_world=frame['footprint_world'])
        width=float(np.linalg.norm(np.array(frame['footprint_world'])[1]-frame['footprint_world'][0])/24)
        ids,positions=objects[frame['map_name']]
        for stride in (2,1):
            world,valid=observed_grid_sites(obs,stride)
            for radius in (2.,1.,.5):
                found=registered_core_proposals(world,valid,positions,ids,radius_m=radius)
                tag=f'n{len(world)}_r{radius:g}'
                for limited in (False,True):
                    if limited and width>4:continue
                    key=tag+('_width4' if limited else '')
                    counters[key]['frames_with_core']+=bool(found)
                    counters[key]['image_instance_pairs']+=len(found)
                    counters[key]['pairs_with_other_instance_negatives']+=len(found) if len(found)>1 else 0
                    groups[key].update((frame['map_name'],oid) for oid in found)
                if stride==2 and radius==1 and width<=4 and len(found)>1:
                    for oid,candidates in found.items():
                        strict.append(dict(frame_key=frame['frame_key'],original_task_key=frame['task_key'],map_name=frame['map_name'],
                            private_training_instance_id=oid,candidate_indices=candidates,
                            available_train_query_indices=registry[frame['map_name']][oid],
                            has_other_instance_core_negatives=True,visibility='unknown',registration='geometry_proxy'))
    args.output.mkdir(parents=True)
    atomic_json(strict,args.output/'links_private.json')
    summary=dict(fit_frames=len(frames),profiles={key:dict(value,distinct_map_instance_groups=len(groups[key])) for key,value in counters.items()},
        selected_profile='n144_r1_width4',selected_pairs=len(strict),
        warning='Geometry support only, not visibility or instance matching. Normal proposal grid/RGB unchanged.',
        source_frames_sha256=file_sha256(args.observations/'public_frames.jsonl'),links_sha256=file_sha256(args.output/'links_private.json'))
    atomic_json(summary,args.output/'summary.json');print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
