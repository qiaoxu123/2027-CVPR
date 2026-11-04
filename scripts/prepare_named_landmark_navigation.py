"""Freeze a public-map-only paired diagnostic from existing direct-relation tasks."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.landmark_alias import canonical_name
from graphnav.landmark_navigation_diagnostic import public_region,oracle_goal_xy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('classified','episode-starts','catalog','data-root','sbf-source','output'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    sys.path.insert(0,str(a.sbf_source))
    from sbfnav.coordinates import MapTransform
    starts={r['index']:r for r in json.loads(a.episode_starts.read_text())}
    objects=json.loads(a.catalog.read_text()); candidates={}; rejected=[]; transforms={}
    named={}
    for mp,items in objects.items():
        named[mp]=defaultdict(list)
        for oid,obj in items.items():
            if obj.get('name'):named[mp][canonical_name(obj['name'])].append((oid,obj))
    for line in a.classified.open():
        r=json.loads(line)
        if r['relation_class']!='direct_geometry':continue
        mp=r['map_name'];start=starts[r['index']]['start_pose']
        if mp not in transforms:
            t=MapTransform.from_raster(a.data_root/'rgbd'/f'{mp}.tif',canvas_size=224,field_size=28)
            transforms[mp]=[t.x_min,t.y_min,t.x_max,t.y_max]
        names={n for path in r['paths'] if path['hops']==1 for n in path['public_names']}
        for name in sorted(names):
            key=mp,canonical_name(name)
            group=named[mp][key[1]]
            try:
                polygons=[obj['contour'] for _,obj in group]
                region=public_region(polygons,transforms[mp])
                goal=oracle_goal_xy(region,start[:2]);distance=float(np.linalg.norm(goal-np.array(start[:2])))
            except ValueError as ex:
                rejected.append(dict(source_index=r['index'],map_name=mp,name=name,reason=str(ex)));continue
            if distance<=20:
                rejected.append(dict(source_index=r['index'],map_name=mp,name=name,reason='start_already_within_polygon20m',distance_m=distance));continue
            types=sorted({obj['object_type'] for _,obj in group})
            category='Building' if types==['Building'] else 'Road' if set(types)<={'TrafficRoad','Footpath','Bridge'} else 'Other'
            task=dict(task_id=f"{mp}::{key[1]}",source_index=r['index'],map_name=mp,landmark_name=name,
                instruction=f'Navigate to {name}.',start_pose=start,public_object_ids=[oid for oid,_ in group],
                public_polygons=polygons,public_types=types,category=category,world_bounds=transforms[mp],
                oracle_goal_xy=goal.tolist(),initial_polygon_distance_m=distance,
                polygon_components=len(list(region.geoms)) if hasattr(region,'geoms') else 1,
                original_instruction=r['instruction'])
            # One physical named union, one start: deterministic, independent of SR/GT.
            priority=hashlib.sha256(f"0:{r['index']}:{key}".encode()).hexdigest()
            if key not in candidates or priority<candidates[key][0]:candidates[key]=(priority,task)
    pool=[v[1] for v in candidates.values()];selected=[]
    for category,quota in [('Building',32),('Road',24),('Other',8)]:
        permap=defaultdict(list)
        for t in pool:
            if t['category']==category:permap[t['map_name']].append(t)
        for mp in permap:permap[mp].sort(key=lambda t:hashlib.sha256(('0:'+t['task_id']).encode()).hexdigest())
        picked=[]
        while len(picked)<quota and any(permap.values()):
            for mp in sorted(permap):
                if permap[mp] and len(picked)<quota:picked.append(permap[mp].pop(0))
        selected.extend(picked)
    if len(selected)<64:
        remaining=sorted([t for t in pool if t not in selected],key=lambda t:hashlib.sha256(('0:'+t['task_id']).encode()).hexdigest())
        selected.extend(remaining[:64-len(selected)])
    if len(selected)!=64:raise ValueError('Insufficient distinct named landmarks for fixed64')
    selected.sort(key=lambda t:(t['map_name'],t['task_id']))
    for i,t in enumerate(selected):t['index']=i
    a.output.mkdir(parents=True)
    atomic_json(selected,a.output/'tasks_public.json')
    atomic_json(rejected,a.output/'selection_exclusions.json')
    atomic_json(dict(tasks=64,categories=dict(Counter(t['category'] for t in selected)),
        maps=len({t['map_name'] for t in selected}),pool_categories=dict(Counter(t['category'] for t in pool)),
        multi_polygon_tasks=sum(len(t['public_object_ids'])>1 for t in selected),
        source_split='val_seen',radius_m=20.,seed=0,task_gt_is_public_name_union=True,
        inference_never_uses_original_target_position=True,
        selection='Map-balanced deterministic public-name groups; begin outside20m; no SR/target coordinate selection',
        tasks_sha256=file_sha256(a.output/'tasks_public.json'),
        inputs_sha256={str(path):file_sha256(path) for path in [a.classified,a.episode_starts,a.catalog,Path(__file__)]}),a.output/'manifest.json')
    print(json.dumps(json.loads((a.output/'manifest.json').read_text()),indent=2))


if __name__=='__main__':main()
