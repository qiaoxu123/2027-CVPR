"""Independently recompute logged post-arrival public geometry; no policy call."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from graphnav.landmark_alias import canonical_name


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    objects=json.loads(a.catalog.read_text())
    rows=[json.loads(x) for x in (a.run/'predictions.jsonl').read_text().splitlines()]
    checks=[]
    for r in rows:
        block=objects[r['map_name']]
        polygons=[Polygon(o['contour']) for o in block.values() if canonical_name(o.get('name',''))==canonical_name(r['landmark_name']) and Polygon(o['contour']).is_valid]
        region=unary_union(polygons)
        for step,d in enumerate(r['step_diagnostics']):
            near=d['near_landmark_geometry']
            if not near['within_20m']:
                assert not near['nearby_objects_after_arrival']
            for item in near['nearby_objects_after_arrival']:
                polygon=Polygon(block[item['public_id']]['contour'])
                dx=polygon.centroid.x-region.centroid.x; dy=polygon.centroid.y-region.centroid.y
                expected=math.degrees(math.atan2(dy,dx))
                error=abs((item['centroid_bearing_deg_from_east']-expected+180)%360-180)
                checks.append(dict(index=r['index'],method=r['method'],step=step,public_id=item['public_id'],
                    distance_error_m=abs(item['polygon_boundary_distance_m']-region.distance(polygon)),
                    bearing_error_deg=error,
                    topology_matches=(item['intersects_public_polygon']==region.intersects(polygon) and item['contains_or_within']==(region.contains(polygon) or polygon.contains(region)))))
    result=dict(checked_entries=len(checks),unique_objects=len({(x['index'],x['public_id']) for x in checks}),
        max_distance_error_m=max(x['distance_error_m'] for x in checks),max_bearing_error_deg=max(x['bearing_error_deg'] for x in checks),
        all_topology_matches=all(x['topology_matches'] for x in checks),all_far_frames_hide_neighbors=True,
        caveat='Directions verified in engineering World XY; geographic north/east calibration is not established by this audit. Public polygon geometry is not visual instance confirmation.')
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
