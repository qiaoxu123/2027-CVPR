"""Post-rollout action/Stop audit, never a policy or success-rule modification."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
from graphnav.closedloop import ClosedLoopProtocol
from scripts.summarize_closedloop import read_run


def pose_constraints(poses,ground,bounds,protocol):
    pose=np.asarray(poses,dtype=np.float64)
    if pose.ndim!=2 or pose.shape[1]!=4 or not np.isfinite(pose).all():
        raise ValueError('finite executed xyzyaw required')
    delta=np.diff(pose[:,:3],axis=0);horizontal=np.linalg.norm(delta[:,:2],axis=-1)
    low=np.asarray(bounds[:2]);high=np.asarray(bounds[2:])
    outside=np.maximum(low-pose[:,:2],0)+np.maximum(pose[:,:2]-high,0)
    distance=np.linalg.norm(outside,axis=-1)
    altitude=pose[:,2]-ground
    violation=np.maximum(protocol.min_altitude_m-altitude,0)+np.maximum(altitude-protocol.max_altitude_m,0)
    # Original invalid starts cannot be changed: only monotonic bounded recovery
    # is allowed until entering the legal interval; no blanket exemption.
    return dict(movements=len(delta),horizontal_step_violations=int((horizontal>protocol.horizontal_step_m+1e-5).sum()),vertical_step_violations=int((abs(delta[:,2])>protocol.vertical_step_m+1e-5).sum()),horizon_violation=int(len(delta)>protocol.horizon),initial_xy_outside=bool(distance[0]>1e-5),initial_altitude_outside=bool(violation[0]>1e-5),xy_outside_worsening=int((distance[1:]>distance[:-1]+1e-5).sum()),altitude_outside_worsening=int((violation[1:]>violation[:-1]+1e-5).sum()),new_illegal_xy=int(((distance[:-1]<=1e-5)&(distance[1:]>1e-5)).sum()),new_illegal_altitude=int(((violation[:-1]<=1e-5)&(violation[1:]>1e-5)).sum()),final_altitude_outside=bool(violation[-1]>1e-5),direction_reversals=int(((delta[1:,:2]*delta[:-1,:2]).sum(-1)<-.8*horizontal[1:]*horizontal[:-1]).sum()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='append',required=True,help='name=completed run')
    for key in ('sbf-source','data-root','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise ValueError('refusing audit overwrite')
    sys.path.insert(0,a.sbf_source)
    from sbfnav.mapdata import GROUND_LEVEL
    from sbfnav.coordinates import MapTransform
    tables=[];summary={};transforms={}
    for spec in a.run:
        name,path=spec.split('=',1);report,rows,manifest=read_run(path)
        protocol=ClosedLoopProtocol(**report['protocol']);counts={};false_distances=[]
        for index,row in rows.items():
            map_name=row['map_name']
            if map_name not in transforms:transforms[map_name]=MapTransform.from_raster(Path(a.data_root)/'rgbd'/(map_name+'.tif'))
            t=transforms[map_name];audit=pose_constraints(row['poses'],GROUND_LEVEL[map_name],(t.x_min,t.y_min,t.x_max,t.y_max),protocol)
            for k,v in audit.items():counts[k]=counts.get(k,0)+v
            if row['metrics']['false_stop']:false_distances.append(row['metrics']['NE'])
            tables.append(dict(model=name,index=index,map_name=map_name,**audit))
        summary[name]=dict(episodes=len(rows),counts=counts,manifest_sha256=manifest,failed_stop_NE_median=float(np.median(false_distances)) if false_distances else None,false_stop_distance_buckets=dict(within20_40=sum(20<d<=40 for d in false_distances),within40_100=sum(40<d<=100 for d in false_distances),beyond100=sum(d>100 for d in false_distances)),note='Source-invalid starts preserved; bounded nonworsening correction is distinguished from new illegal states. No swept 3D collision, perspective visibility or physical flight-safety guarantee.')
    out.mkdir(parents=True);(out/'summary.json').write_text(json.dumps(summary,indent=2))
    with (out/'episodes.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(tables[0]));w.writeheader();w.writerows(tables)
    print(json.dumps(summary))


if __name__=='__main__':main()
