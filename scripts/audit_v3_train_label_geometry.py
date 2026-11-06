"""TRAIN-only private-label consistency; not visibility or pixel-instance accuracy."""
import argparse,json
from pathlib import Path
from collections import Counter
import numpy as np
from shapely.geometry import Point,Polygon
from graphnav.checkpointing import atomic_json,file_sha256

def main():
    p=argparse.ArgumentParser()
    for k in ['train-jsonl','objects','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();objects=json.loads(a.objects.read_text());rows=[];physical=set();counts=Counter();types={}
    for line in a.train_jsonl.read_text().splitlines():
        r=json.loads(line);parts=r['episode_id'].split(':');key=(r['map_name'],parts[1]);physical.add(key);obj=objects.get(key[0],{}).get(key[1]);bb=np.asarray(r['world_bounds']);gt=bb[:2]+np.asarray(r['target_xy'])*(bb[2:]-bb[:2]);category=obj.get('object_type','Unknown') if obj else 'Unknown'
        stat=types.setdefault(category,Counter());counts['queries']+=1;stat['queries']+=1
        if obj is None:
            counts['missing_object']+=1;stat['missing_object']+=1;continue
        contour=np.asarray(obj.get('contour',[]),dtype=float)
        if contour.ndim!=2 or contour.shape[1]!=2 or len(contour)<3 or not np.isfinite(contour).all():counts['invalid_contour']+=1;continue
        polygon=Polygon(contour)
        if not polygon.is_valid or polygon.area<=0:counts['invalid_polygon']+=1;continue
        distance=float(polygon.distance(Point(gt)));object_xy=np.asarray(obj['position'],dtype=float)[:2];offset=float(np.linalg.norm(gt-object_xy))
        for name,flag in [('point_outside_contour_1mm',distance>.001),('point_contour_distance_gt20m',distance>20),('point_registration_offset_gt1m',offset>1),('point_registration_offset_gt20m',offset>20)]:
            counts[name]+=int(flag);stat[name]+=int(flag)
        if distance>20 or offset>1:rows.append(dict(episode_id=r['episode_id'],category=category,point_to_contour_m=distance,point_vs_registered_position_m=offset))
    atomic_json(dict(summary=dict(counts),physical_targets=len(physical),by_category={k:dict(v) for k,v in types.items()},exceptions=rows,train_sha256=file_sha256(a.train_jsonl),objects_sha256=file_sha256(a.objects),scope='TRAIN label geometry audit only; no filtering, model inference, visibility, pixel GT or weight changes'),a.output);print(json.dumps(dict(counts),indent=2))
if __name__=='__main__':main()
