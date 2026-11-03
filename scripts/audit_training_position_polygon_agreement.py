"""TRAIN-fit annotation consistency only; never claims Contour is pixel truth."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon,Point
from graphnav.checkpointing import atomic_json,file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','fit-records','catalog','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    fit=json.loads(a.fit_records.read_text());wanted={x['frame_key'] for x in fit};catalog=json.loads(a.catalog.read_text());targets={}
    for obs in ('fixed_yaw_train400_s1','scoped_between_train294_observations_v1'):
        for x in map(json.loads,(a.root/obs/'labels_private.jsonl').read_text().splitlines()):
            if x.get('frame_key') not in wanted:continue
            mp,oid=x['physical_group'];key=(mp,str(oid))
            if key in targets:continue
            obj=catalog[mp][str(oid)];contour=np.asarray(obj['contour'],float);valid=False;distance=None
            if contour.ndim==2 and contour.shape[1]==2 and len(contour)>=3 and np.isfinite(contour).all():
                poly=Polygon(contour);valid=poly.is_valid and poly.area>0
                if valid:distance=Point(x['target_world'][:2]).distance(poly)
            targets[key]=dict(task_key=x['task_key'],map_name=mp,target_id=str(oid),category=x['category'],
                valid_polygon=bool(valid),position_to_contour_distance_m=distance)
    summary=dict(target_id_groups=len(targets),by_category={},
        warning='Single object ID can describe a group of vehicles. Position/Contour agreement is annotation consistency, not RGB visibility or pixel identity.',
        normal_inference_modified=False,used_for_training=False,
        sha256={k:file_sha256(v) for k,v in dict(fit_records=a.fit_records,catalog=a.catalog,producer=Path(__file__)).items()})
    for c in ('Car','Building','Parking','Ground'):
        rr=[x for x in targets.values() if x['category']==c];d=[x['position_to_contour_distance_m'] for x in rr if x['valid_polygon']]
        summary['by_category'][c]=dict(target_id_groups=len(rr),valid_polygons=len(d),invalid_polygons=len(rr)-len(d),
            outside=sum(v>.01 for v in d),distance_over2m=sum(v>2 for v in d),distance_over5m=sum(v>5 for v in d),
            distance_over20m=sum(v>20 for v in d),max_distance_m=max(d) if d else None)
    a.output.mkdir(parents=True);atomic_json(list(targets.values()),a.output/'records_private.json');atomic_json(summary,a.output/'summary.json');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
