"""Posthoc identity/coordinate agreement audit; never used for feature selection."""
import argparse,json,collections
import numpy as np
from shapely.geometry import Polygon,Point

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--public-objects',required=True);p.add_argument('--output',required=True);a=p.parse_args();rows=json.load(open(a.manifest))['records'];catalog=json.load(open(a.public_objects));counts=collections.Counter();distances=[]
 for row in rows:
  obj=catalog[row['map_name']].get(str(row['episode_id']).split(':')[1])
  if obj is None:counts['missing']+=1;continue
  bounds=np.asarray(row['world_bounds']);gt=bounds[:2]+np.asarray(row['target_xy'])*(bounds[2:]-bounds[:2]);poly=Polygon(obj['contour'])
  if not poly.is_valid:counts['invalid']+=1;continue
  distance=poly.distance(Point(gt));distances.append(distance);counts['within1m' if distance<=1 else 'outside1m']+=1
 result=dict(counts=counts,max_distance_m=max(distances),percentiles_m=np.percentile(distances,[50,90,99]).tolist(),interpretation='Coordinate/ID disagreement is uncertain weak supervision; it does not establish which field is wrong. Primary angle-only training never uses instance-ID loss.')
 with open(a.output,'w') as f:json.dump(result,f,indent=2)
 print(json.dumps(result))
if __name__=='__main__':main()
